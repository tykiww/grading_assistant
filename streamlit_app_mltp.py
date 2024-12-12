import re
import time
import concurrent.futures
from math import floor
from io import BytesIO

import streamlit as st
import pandas as pd
from openai import OpenAI
import plotly.graph_objs as go

class GradingAssistant:
    def __init__(self):
        # Initialize session state variables
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize and reset session state variables."""
        state_defaults = {
            "grading_started": False,
            "qa_data": [],
            "qa_df": pd.DataFrame(columns=[
                "Question #", "Context", "Question", "Grading Guidance", 
                "Exemplary Answer", "Total Points", "Point Allocation Guide"
            ]),
            "grading_results": None,
            "form_reset": False
        }

        for key, default in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    def verify_openai_api_key(self, key):
        """Verify OpenAI API key format."""
        return key and key.startswith("sk-") and len(key) > 20

    def clean_file(self, df, file_type='answer'):
        """Clean uploaded files, removing empty rows and columns."""
        # Remove empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')

        if file_type == 'grading':
            df = self._clean_grading_columns(df)
        
        return df

    def _clean_grading_columns(self, df):
        """Clean column names in grading file."""
        def clean_column_name(col, last):
            col = re.sub(r'\n+', ' ', col).strip().replace(":", "")
            
            # Extract 'Question N'
            match = re.search(r'Question\s*(\d+)', col)
            if match:
                col = f'Question {match.group(1)}'
            
            # Handle numeric columns
            try:
                col = float(col)
                col = floor(col)
                col = f"{last} Points ({col})"
            except ValueError:
                pass

            return col
        
        last = ''
        store = []
        for col in df.columns:
            retain = clean_column_name(col, last)
            store.append(retain)
            last = retain

        df.columns = store
        return df

    def setup_grading_query(self, student_answer, question_details):
        """Prepare prompt for AI grading."""
        return f"""Given requirements, grading guidance, the question of interest, and exemplary answer, properly score the student answer:
# Requirements
Do not hallucinate.
Do not bring outside knowledge.
Provide a point score (out of {question_details['Total Points']}) and concise justification for the score.
Find every reason to give points rather than to remove points.
Make sure that your final score is only a number.
Be exact in your point allocation.

Format your score response as:
Assessment: [justification with point scores]
Score: [initial number]
Reassessment of reason and score: [re-evaluation]
Revised Score: [final number]

# Guidance
Grading Guidance: {question_details['Grading Guidance']}
Total Points Possible: {question_details['Total Points']}
Point Allocation Guide: {question_details['Point Allocation Guide']}

# Question
Context: {question_details['Context']}
Question: {question_details['Question']}

# Exemplary Answer
{question_details['Exemplary Answer']}

# Student Answer
{student_answer}

# Score
"""

    def query_openai(self, prompt, api_key):
        """Query OpenAI for grading."""
        client = OpenAI(api_key=api_key)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": "You are an experienced educational grader with expertise in providing detailed, fair assessments of student work."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error querying OpenAI: {str(e)}")
            return None

    def clean_output(self, response):
        """Clean and extract score and justification from AI response."""
        try:
            justification_part = response.split('Assessment:')[1].split('Revised Score:')[0].strip()
            score_part = response.split('Revised Score:')[1].strip()
            
            score = float(score_part)
            justification = justification_part.strip()
            
            return score, justification
        except Exception as e:
            st.error(f"Error parsing AI response: {str(e)}")
            return 0, "Unable to parse grading response"

    def grade_single_submission(self, submission, answer_key, api_key):
        """Grade a single student's submission."""
        results = []
        for _, row in answer_key.iterrows():
            # Find matching question
            question_col = f"Question {row['Question #']}"
            points_col = f"Question {row['Question #']} Points"
            
            response = submission[question_col]
            points = submission.get(points_col, 0)

            # Prepare grading query
            prompt = self.setup_grading_query(
                response,
                row.to_dict()
            )

            # Query OpenAI for grading
            grading_response = self.query_openai(prompt, api_key)
            
            if grading_response:
                new_score, justification = self.clean_output(grading_response)

                result = {
                    'student_name': submission.get('name', 'Unknown'),
                    'student_id': submission.get('id', 'Unknown'),
                    'q_num': row['Question #'],
                    'old_score': points,
                    'new_score': new_score,
                    'assessment': justification,
                    'context': row['Context'],
                    'question': row['Question'],
                    'answer': response,
                    'exemplary': row['Exemplary Answer'], 
                }
                results.append(result)

        return results

    def parallel_grade_submissions(self, student_submissions, answer_key, api_key, max_workers=4):
        """Parallelize grading across submissions."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Prepare submission grading tasks
            futures = {
                executor.submit(self.grade_single_submission, submission, answer_key, api_key): 
                submission 
                for _, submission in student_submissions.iterrows()
            }

            # Track progress
            total = len(futures)
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    submission_results = future.result()
                    results.extend(submission_results)
                    
                    # Optional: Update progress (you might want to integrate this with Streamlit)
                    print(f"Processed submission {i}/{total}")
                except Exception as e:
                    print(f"Error processing submission: {e}")

        return results

    def build_boxplots(self, data):
        """Create boxplots for score comparisons."""
        fig_boxplots = go.Figure()
        fig_boxplots.add_trace(go.Box(
            y=data['old_score'],
            x=data['q_num'],
            name='Old Score',
            marker_color='blue',
            boxmean='sd',
            jitter=0.3,
            pointpos=-1.5
        ))

        fig_boxplots.add_trace(go.Box(
            y=data['new_score'],
            x=data['q_num'],
            name='New Score',
            marker_color='orange',
            boxmean='sd',
            jitter=0.3,
            pointpos=1.5
        ))

        fig_boxplots.update_layout(
            title="Old vs New Scores by Question Distribution",
            xaxis_title="Question Number",
            yaxis_title="Score",
            boxmode='group'
        )
        return fig_boxplots

    def run_app(self):
        """Main Streamlit application."""
        st.title("📊 AI Grading Assistant")

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Upload Files", "Grading Results", "Question & Answer Builder"])

        with tab1:
            self._upload_and_grade_tab()

        with tab2:
            self._results_tab()

        with tab3:
            self._qa_builder_tab()

    def _upload_and_grade_tab(self):
        """Handle file uploads and grading process."""
        st.write(
            "Follow the steps below: \n"
            "1) Copy and paste your OpenAI API Key.\n"
            "2) Upload student submission CSV and inspect the file data.\n"
            "3) Upload answer key CSV and inspect the file data.\n"
            "4) Hit the 'Start Grading' button to initiate grading."
        )

        # OpenAI API Key input
        openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")

        if not openai_api_key or not self.verify_openai_api_key(openai_api_key):
            st.info("Please provide a valid OpenAI API key.", icon="🗝️")
            return

        st.success("API key added successfully!", icon="🔓")

        # File uploads
        submissions_file = st.file_uploader("📄 Upload student submissions (CSV)", type=["csv"])
        answer_key_file = st.file_uploader("📄 Upload answer key and instructions (CSV)", type=["csv"])

        if submissions_file and answer_key_file:
            # Preview files
            student_submissions = pd.read_csv(submissions_file)
            answer_key = pd.read_csv(answer_key_file)

            student_submissions = self.clean_file(student_submissions, 'grading')
            answer_key = self.clean_file(answer_key, 'answer')

            st.write("### Student Submissions Preview:")
            st.dataframe(student_submissions.head())

            st.write("### Answer Key Preview:")
            st.dataframe(answer_key.head())

            # Grading controls
            max_rows = len(student_submissions)
            data_length = st.number_input("For Testing: Enter a number to truncate your student file", 
                                           min_value=1, max_value=max_rows, value=max_rows)

            if st.button("Start Grading"):
                # Truncate data for testing
                truncated_data = student_submissions.head(data_length)

                # Start grading process
                start_time = time.time()
                results = self.parallel_grade_submissions(
                    truncated_data, 
                    answer_key, 
                    openai_api_key
                )

                # Store and display results
                st.session_state['grading_started'] = True
                st.session_state['grading_results'] = results
                st.success(f"Grading completed in {time.time() - start_time:.2f} seconds!")

    def _results_tab(self):
        """Display grading results."""
        st.title("Grading Results")

        if not st.session_state.get("grading_started", False):
            st.write("No grading has started yet. Please upload files and start grading.")
            return

        results = st.session_state.get('grading_results', [])
        if not results:
            st.write("No results found.")
            return

        # Convert results to DataFrame
        data = pd.DataFrame(results)

        # Boxplot visualization
        st.write("### Grading Distribution")
        fig_boxplots = self.build_boxplots(data)      
        st.plotly_chart(fig_boxplots)

        # Editable results table
        st.write("### Grading Breakdown")
        column_configuration = {
            "Student Name": st.column_config.TextColumn("Student Name"),
            "Student ID": st.column_config.NumberColumn("Student ID"),
            "Question #": st.column_config.NumberColumn("Question #", min_value=0),
            "Old Score": st.column_config.NumberColumn("Old Score", min_value=0),
            "New Score": st.column_config.NumberColumn("New Score", min_value=0),
            "Assessment": st.column_config.TextColumn("Assessment"),
            "Context": st.column_config.TextColumn("Context"),
            "Question": st.column_config.TextColumn("Question"),
            "Answer": st.column_config.TextColumn("Student Answer"), 
            "Exemplary": st.column_config.TextColumn("Exemplary"),
        }

        editable_df = st.data_editor(
            data,
            column_config=column_configuration,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed"
        )

    def _qa_builder_tab(self):
        """Question and Answer builder tab."""
        st.title("Answer Key Builder")
        st.write("Enter the details for each question below.")

        # Input form
        with st.form("Add Q&A Form", clear_on_submit=True):
            question_number = st.text_input("Question #")
            context = st.text_area("Context for Essay Question")
            question = st.text_area("Question")
            grading_guidance = st.text_area("Grading Guidance")
            exemplary_answer = st.text_area("Exemplary Answer")
            total_points = st.number_input("Total Points", min_value=1, value=1)
            point_allocation = st.text_area("Point Allocation Guide")

            submitted = st.form_submit_button("Add Question")
            
            if submitted:
                new_row = pd.DataFrame({
                    "Question #": [question_number],
                    "Context": [context],
                    "Question": [question],
                    "Grading Guidance": [grading_guidance],
                    "Exemplary Answer": [exemplary_answer],
                    "Total Points": [total_points],
                    "Point Allocation Guide": [point_allocation]
                })
                
                st.session_state['qa_df'] = pd.concat([
                    st.session_state['qa_df'], 
                    new_row
                ], ignore_index=True)

        # Display and edit existing Q&A table
        st.write("### Current Q&A Table")
        editable_df = st.data_editor(
            st.session_state['qa_df'],
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )

        # Update the DataFrame in session state
        st.session_state['qa_df'] = editable_df

        # Download button
        if not st.session_state['qa_df'].empty:
            csv_data = st.session_state['qa_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Q&A as CSV",
                data=csv_data,
                file_name="qa_answer_key.csv",
                mime='text/csv'
            )

def main():
    grading_assistant = GradingAssistant()
    grading_assistant.run_app()

if __name__ == "__main__":
    main()