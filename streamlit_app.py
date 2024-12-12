## TODO: 
### Progress bar.
### 0.15/1000000 input * 1195 * 122 (2.2 cents)
### 0.075/1000000 output * 75 * 122 (.7 cents)

### Dashboard with 2 views. 1 overall by question. Statistically significant or not through bootstrapping.
### Table view of Student ID, Question #, Their Answer, Original score, New Score, justification
### Table should have ability to change grade and have wide rows.
### Ability to export file, but when exporting, including student name & all other original answers.

## Consider: 
### Re-converting student answers to an amenable format (tables suck)
### Counting is terrible, so may need to re-engineer by looping, or multi-shotting with human feedback, or bare-bones answer finding.
### May need to include exhibit figures as interpretation points.

############################### IMPORTS ###############################

import re
import time
from math import floor
from io import BytesIO

import streamlit as st
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import plotly.graph_objs as go

################################# SETUP #################################

# Function to verify OpenAI API key
def verify_openai_api_key(key):
    if not key:
        return False
    return key.startswith("sk-") and len(key) > 20

# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to clean the answer file
def clean_answer_file(df):
    # Remove any empty rows
    df = df.dropna(how='all')
    # Remove any empty columns
    df = df.dropna(axis=1, how='all')
    return df

# Function to clean the grading file
def clean_grading_file(df):
    def clean_column_name(col, last):
        # Remove line breaks, leading/trailing spaces, and colons
        col = re.sub(r'\n+', ' ', col).strip()
        col = col.replace(":", "").strip()

        # Extract 'Question N' if present
        match = re.search(r'Question\s*(\d+)', col)
        if match:
            col = f'Question {match.group(1)}'
        
        # Extract just numbers
        try:
            # Try converting the column name to a float, then floor it.
            col = float(col)
            col = floor(col)
            # Append the last column if it is just a pure number.
            col = last + ' Points (' + str(col) + ')'
        except ValueError:
            pass  # Skip if it's not a number

        return col
    
    # Remove any empty rows
    df = df.dropna(how='all')
    # Remove any empty columns
    df = df.dropna(axis=1, how='all')

    # Apply cleaning function to all column names
    last = ''
    store = []
    for col in df.columns:
       retain = clean_column_name(col, last)
       store.append(retain)
       last = retain

    df.columns = store

    return df


# Function to set up query to OpenAI
def setup_grading_query(student_answer, question, context, exemplar_answer, grading_guidance, total_points, point_allocation):
    return f"""Given requirements, grading guidance, the question of interest, and exemplary answer, properly score the student answer:
# Requirements
Do not hallucinate.
Do not bring outside knowledge.
Provide a point score (out of {total_points}) and concise justification for the score.
Find every reason to give points rather than to remove poiints.
Make sure that your final score is only a number.
Be exact in your point allocation.

Format your score response as:
Assessment: [justification with point scores]
Score: [final number]


# Guidance
Grading Guidance: {grading_guidance}
Total Points Possible: {total_points}
Point Allocation Guide: {point_allocation}

# Question
Context: {context}
Question: {question}

# Exemplary Answer
{exemplar_answer}

# Student Answer
{student_answer}

# Score
"""

def match_answers(number, student_response):
    # Find question
    r_idx = "Question " + str(number)
    r_idx = student_response.index.get_loc(r_idx)
    p_idx = r_idx + 1
    return student_response[r_idx], student_response[p_idx]



# Function to query OpenAI for answers
def query_openai(prompt, api_key):

    client = OpenAI(api_key=api_key)

    try:
        response = chat_completion = client.chat.completions.create(
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


# Clean Output
def clean_output(response):
    # Split by the Score: and Justification: markers
    justification_part = response.split('Assessment:')[1].split('Score:')[0].strip()
    score_part = response.split('Score:')[1].strip()
    
    # Extract score and justification
    score = float(score_part)
    justification = justification_part.strip()
    
    return score, justification



# Function to calculate grading for individual
def grade_submission(student_response, answer_key, api_key):
    results = []
    for _, row in answer_key.iterrows():

        # match answer key to student answer.
        response, points = match_answers(row['Question #'], student_response)

        # Prepare prompt
        prompt = setup_grading_query(
            response,
            row['Question'],
            row['Context'],
            row['Exemplary Answer'],
            row['Grading Guidance'],
            row['Total Points'],
            row['Point Allocation Guide']
        )

        # Prompt OpenAI
        grading_response = query_openai(prompt, api_key)

        # Clean prompt
        new_score, justification = clean_output(grading_response)

        # Prepare outputs
        if grading_response:

            output = {
                'student_id': student_response['id'],
                'q_num': row['Question #'],
                'old_score': points,
                'new_score': new_score,
                'assessment': justification,
                'context': row['Context'],
                'question': row['Question'],
                'answer': response,
                'exemplary': row['Exemplary Answer'], 
            }

            results.append(output)

    return results


def build_boxplots(data):
    fig_boxplots = go.Figure()
    fig_boxplots.add_trace(go.Box(
        y=data['old_score'],
        x=data['q_num'],
        name='Old Score',
        marker_color='blue',
        boxmean='sd',  # Shows the mean and standard deviation
        jitter=0.3,  # Adds jitter to the points
        pointpos=-1.5  # Positions points to the left of the box
    ))

    fig_boxplots.add_trace(go.Box(
        y=data['new_score'],
        x=data['q_num'],
        name='New Score',
        marker_color='orange',
        boxmean='sd',  # Shows the mean and standard deviation
        jitter=0.3,  # Adds jitter to the points
        pointpos=1.5  # Positions points to the right of the box
    ))

    # Update the layout for better visualization
    fig_boxplots.update_layout(
        title="Old vs New Scores by Question Distribution",
        xaxis_title="Question Number",
        yaxis_title="Score",
        boxmode='group'  # Group the boxplots together
    )
    return fig_boxplots

################################# STATES #################################

# Initialize session states
if "grading_started" not in st.session_state:
    st.session_state["grading_started"] = False

if 'qa_data' not in st.session_state:
    st.session_state['qa_data'] = []

if 'qa_df' not in st.session_state:
    st.session_state['qa_df'] = pd.DataFrame(columns=[
        "Question #", "Context", "Question", "Grading Guidance", 
        "Exemplary Answer", "Total Points", "Point Allocation Guide"
    ])

if 'grading_results' not in st.session_state:
    st.session_state['grading_results'] = None

# Initialize form reset trigger
if 'form_reset' not in st.session_state:
    st.session_state.form_reset = False

# Function to handle form submission
def handle_form_submit():
    # Create new row from current values
    new_row = pd.DataFrame({
        "Question #": [st.session_state.q_number_input],
        "Context": [st.session_state.context_input],
        "Question": [st.session_state.question_input],
        "Grading Guidance": [st.session_state.grading_guidance_input],
        "Exemplary Answer": [st.session_state.exemplary_answer_input],
        "Total Points": [st.session_state.total_points_input],
        "Point Allocation Guide": [st.session_state.point_allocation_input],
    })
    
    # Append the new row to the DataFrame
    st.session_state['qa_df'] = pd.concat([st.session_state['qa_df'], new_row], ignore_index=True)
    
    # Clear the form fields in session state
    st.session_state.q_number_input = ""
    st.session_state.context_input = ""
    st.session_state.question_input = ""
    st.session_state.grading_guidance_input = ""
    st.session_state.exemplary_answer_input = ""
    st.session_state.total_points_input = 1
    st.session_state.point_allocation_input = ""

################################# APP #################################

# Create tabs for the operations
tab1, tab2, tab3 = st.tabs(["Upload Files", "Grading Results", "Question & Answer Builder"])

with tab1:
    st.title("📊 Grading Assistant")
    st.write(
        "Follow the steps below: \n"
        "1) Copy and paste your OpenAI API Key.\n"
        "2) Upload student submission CSV and inspect the file data.\n"
        "3) Upload answer key CSV and inspect the file data.\n"
        "4) Hit the 'Start Grading' button to initiate grading."
    )

    # Ask user for their OpenAI API key
    openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")

    if not openai_api_key:
        st.info("Please provide your OpenAI API key to continue.", icon="🗝️")
    elif not verify_openai_api_key(openai_api_key):
        st.error("Invalid API key.")
    else:
        st.success("API key added successfully!", icon="🔓")

        # Upload student submissions
        submissions_file = st.file_uploader(
            "📄 Upload student submissions (CSV)", type=["csv"]
        )

        if submissions_file:
            st.write("### Student Submissions Preview:")
            student_submissions = pd.read_csv(submissions_file) # submissions_file = 'data/student_answers.csv'
            student_submissions = clean_grading_file(student_submissions)

            if "name" in student_submissions.columns:
                temp = student_submissions.drop(["name"],axis=1).head()
                st.dataframe(temp)
            else:
                st.dataframe(student_submissions.head())

        # Upload answer key
        answer_key_file = st.file_uploader(
            "📄 Upload answer key and instructions (CSV)", type=["csv"]
        )

        if answer_key_file:
            st.write("### Answer Key and Instructions Preview:")
            answer_key = pd.read_csv(answer_key_file) # answer_key_file = 'data/answer_key.csv'
            answer_key = clean_answer_file(answer_key)
            st.dataframe(answer_key.head())

        # Start grading process
        if submissions_file and answer_key_file:

            max_rows = len(student_submissions)
            data_length = st.number_input("For Testing: Enter a number to truncate your student file", min_value=1, max_value=max_rows, value=max_rows)

            if st.button("Start Grading"):
                st.session_state["grading_started"] = True
                st.write("🚧 Grading in progress...")
                
                # Truncate data for testing
                truncated_data = student_submissions[0:data_length] 

                # Progress bar, status, and timer
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                total = len(truncated_data)
                start_time = time.time()

                # Process each submission
                results = []
                for i, submission in truncated_data.iterrows():

                    # Actual work (call your grading function)
                    student_grades = grade_submission(submission, answer_key, openai_api_key)
                    results.extend(student_grades)

                    # Update the progress bar (use percentage based on i and total)
                    progress_bar.progress(int((i + 1) / total * 100))  # Progress in percentage

                    # Timer calculation
                    elapsed_time = time.time() - start_time
                    estimated_total_time = (elapsed_time / (i + 1)) * total  # Estimate total time
                    remaining_time = estimated_total_time - elapsed_time
                    remaining_minutes = remaining_time / 60

                    # Update status and time text
                    status_text.text(f'Progress: {int((i + 1) / total * 100)}%')
                    time_text.text(f'Estimated Time to Completion: {remaining_minutes:.2f} minutes')

                # When complete, mark the progress as 100% and show a success message
                progress_bar.progress(100)

                # Store results in session state
                st.session_state['grading_results'] = results
                st.success("Grading completed!")

with tab2:

    st.title("Grading Results")

    if st.session_state["grading_started"]:
        if st.session_state['grading_results']:
            st.write("### Grading Distribution")

            # Retrieve graded results.
            data = pd.DataFrame(st.session_state['grading_results'])
                #data = pd.DataFrame(results)

            # Score distribution by grade
            fig_boxplots = build_boxplots(data)      
            st.plotly_chart(fig_boxplots)

            st.write("### Grading Breakdown")

            column_configuration = {
                "Student ID": st.column_config.TextColumn("Student ID"),
                "Question #": st.column_config.NumberColumn("Question #", min_value=0),
                "Old Score ": st.column_config.NumberColumn("Old Score", min_value=0),
                "New Score": st.column_config.NumberColumn("New Score", min_value=0),
                "Assessment": st.column_config.TextColumn("Assessment"),
                "Context": st.column_config.TextColumn("Context"),
                "Question": st.column_config.TextColumn("Question"),
                "Student Answer": st.column_config.TextColumn("Student Answer"), 
                "Exemplary": st.column_config.TextColumn("Exemplary"),
            }


            editable_df = st.data_editor(
                data,
                column_config=column_configuration,
                use_container_width=False,
                hide_index=True,
                num_rows="fixed",
                
            )

    else:
        st.write("No grading has started yet. Please upload the files and start grading from the first tab.")

with tab3:
    st.title("Answer Key Builder")
    st.write("Enter the details for each question below, then download or submit the answer key.")

    # Input form with proper state management
    with st.form("Add Q&A Form", clear_on_submit=True):
        st.text_input("Question #", key='q_number_input')
        st.text_area("Context for Essay Question", key='context_input')
        st.text_area("Question", key='question_input')
        st.text_area("Grading Guidance", key='grading_guidance_input')
        st.text_area("Exemplary Answer", key='exemplary_answer_input')
        st.number_input("Total Points", min_value=1, value=1, key='total_points_input')
        st.text_area("Point Allocation Guide", key='point_allocation_input')

        submitted = st.form_submit_button("Add Question", on_click=handle_form_submit)


    # Reset form_reset flag after form is rendered
    if st.session_state.get('form_reset', False):
        st.session_state.form_reset = False

    # Display and edit existing Q&A table
    st.write("### Current Q&A Table")
    column_configuration = {
        "Question #": st.column_config.TextColumn("Question #"),
        "Context": st.column_config.TextColumn("Context"),
        "Question": st.column_config.TextColumn("Question"),
        "Grading Guidance": st.column_config.TextColumn("Grading Guidance"),
        "Exemplary Answer": st.column_config.TextColumn("Exemplary Answer"),
        "Total Points": st.column_config.NumberColumn("Total Points", min_value=1),
        "Point Allocation Guide": st.column_config.TextColumn("Point Allocation Guide"),
    }

    editable_df = st.data_editor(
        st.session_state['qa_df'],
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic"
    )

    # Update the DataFrame in session state
    st.session_state['qa_df'] = editable_df

    # Download and submit buttons
    if not st.session_state['qa_df'].empty:
        csv_data = convert_df_to_csv(st.session_state['qa_df'])
        st.download_button(
            "📥 Download Q&A as CSV",
            data=csv_data,
            file_name="qa_answer_key.csv",
            mime='text/csv'
        )

        if st.button("Submit Q&A for Grading"):
            st.session_state['qa_data'] = st.session_state['qa_df'].copy()
            st.success("Q&A submitted for use in grading!")