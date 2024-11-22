from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import time
import json
import requests
import pandas as pd
import streamlit as st
import PyPDF2

load_dotenv()
default_llm = AzureChatOpenAI(
    azure_endpoint='',
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    temperature=0
)

os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
os.environ['MODEL_NAME'] = 'Azure'
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_API_VERSION'] = '2023-08-01-preview'
os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_EMBEDDINGS')



# Extract vacacny, Skillsets, experince, education qulifaction
def get_job_description(text):
    job_description_parser_agent = Agent(
        role="Job Description Analyser",
        goal="Accurately analyse the given job description to find out the required skills, experience and other details.",
        verbose=True,
        llm=default_llm,
        allow_delegation=False,
        backstory="""
        You are a skilled Job Description Analyser. You have been working as Human Resource Manager for a long time.
        You have helped hired hundreds of employees and have a good understanding of different job descriptions.
        You can easily identify the required skills, experience and other details from the given job description.
        You have to correctly identify the given fields:
            1) Job Role
            2) Skillsets
            3) Experience
            4) Education_Qualification
            5) Preferred_Skill
            6) Responsibilities
            7) Location
    """)

    job_description_parsing_task = Task(
        description=(
            f"""
                    ```{text}```

                    From the job_desription given above, identify the given fields:
                    1) Job Role
                    2) Skillsets
                    3) Experience
                    4) Education_Qualification
                    5) Preferred_Skill
                    6) Responsibilities
                    7) Location

                    Make sure to follow the given set of rules
                    1) Perform a detailed analysis of the context and extract the required fields.
                    2) Make sure that no data is missed out and the fields are accurately extracted.
                    3) Make sure these are the only fields in the output as the following process will not work with any other fields.
                    4) In location provide only - city or state or country, like where the job location is
                    5) Provide the output in the json format.

                    Do not provide any other texts or information or symbols like ``` in the output as it will not work with the further process.
                    Make sure the provide the output in the json format.
                    Important note: Every step mentioned above must be followed to get the required results.
                    Do not provide any other texts or information in the output as it will not work with the further process.
                    Do not include ``` or any other such characters in the output.
                """
        ),
        agent=job_description_parser_agent,
        expected_output="A json with the values of the required fields",
    )

    jd_parser_crew = Crew(
        agents=[
            job_description_parser_agent
        ],
        tasks=[
             job_description_parsing_task
        ],
        verbose=True,
        allow_delegation=False,
        cache=True
    )

    result = jd_parser_crew.kickoff()

    return result

# Format returned by 1st proxycurl
def get_employees_data():
    return {
        "employees": [
            {
                "profile_url": "https://www.linkedin.com/in/prasant-bagale/",
                "profile": None,
                "last_updated": "2024-09-06T17:41:19Z"
            }
        ]

    }

# Format returned by proxycurl for 1 person (by his/her linkedin url)
def get_employee_data_hardcoded():
    return {
        "public_identifier": "pragya-singh-828b1521b",
        "profile_pic_url": "https://media.licdn.com/dms/image/v2/D4D03AQGXhtr20QdcHg/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1729188521430?e=1737590400&v=beta&t=sfzh0cpRBwpdSG46zSOcN1KRSwHfVffH4MvRuBzHmCc",
        "background_cover_image_url": None,
        "first_name": "Pragya",
        "last_name": "Singh",
        "full_name": "Pragya Singh",
        "follower_count": 2346,
        "occupation": "AI/ML engineer  at Cyncly",
        "headline": "AI/ML engineer @Cyncly | BS student at IIT Madras",
        "summary": None,
        "country": "IN",
        "country_full_name": "India",
        "city": "Bengaluru",
        "state": "Karnataka",
        "experiences": [
            {
                "starts_at": {
                    "day": 1,
                    "month": 10,
                    "year": 2024
                },
                "ends_at": None,
                "company": "Cyncly",
                "company_linkedin_profile_url": "https://www.linkedin.com/company/cyncly",
                "company_facebook_profile_url": None,
                "title": "AI/ML engineer ",
                "description": None,
                "location": "Bengaluru, Karnataka, India",
                "logo_url": "https://s3.us-west-000.backblazeb2.com/proxycurl/company/cyncly/profile?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0004d7f56a0400b0000000001%2F20241122%2Fus-west-000%2Fs3%2Faws4_request&X-Amz-Date=20241122T034049Z&X-Amz-Expires=1800&X-Amz-SignedHeaders=host&X-Amz-Signature=77f140cee0fa62b1173332786de158647caea451ebb7c3061289a9d8b0d372c4"
            },
            {
                "starts_at": {
                    "day": 1,
                    "month": 1,
                    "year": 2024
                },
                "ends_at": {
                    "day": 31,
                    "month": 10,
                    "year": 2024
                },
                "company": "Fiery",
                "company_linkedin_profile_url": "https://www.linkedin.com/company/fieryprint",
                "company_facebook_profile_url": None,
                "title": "Data Science Intern",
                "description": "Worked on various projects like Building a RAG application on company's help documentation, intent recognition app and a home decor app.",
                "location": "Bengaluru, Karnataka, India",
                "logo_url": "https://s3.us-west-000.backblazeb2.com/proxycurl/company/fieryprint/profile?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0004d7f56a0400b0000000001%2F20241122%2Fus-west-000%2Fs3%2Faws4_request&X-Amz-Date=20241122T034049Z&X-Amz-Expires=1800&X-Amz-SignedHeaders=host&X-Amz-Signature=7937dd3701d9ec4017fc77034d6bf1a27a2752c73033121868fae6733ec4090f"
            },
            {
                "starts_at": {
                    "day": 1,
                    "month": 10,
                    "year": 2023
                },
                "ends_at": {
                    "day": 30,
                    "month": 11,
                    "year": 2023
                },
                "company": "NutriNation",
                "company_linkedin_profile_url": "https://www.linkedin.com/company/90543004/",
                "company_facebook_profile_url": None,
                "title": "Data Science Intern",
                "description": None,
                "location": "Chennai, Tamil Nadu, India",
                "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQHP4F1BR4ijUw/company-logo_400_400/company-logo_400_400/0/1685522942237/cook_ai_logo?e=1740614400&v=beta&t=AX_EDOznlUQhDEolqxiLtJL0KavJ6pLLJrGvCQKT7XU"
            }
        ],
        "education": [
            {
                "starts_at": {
                    "day": 1,
                    "month": 9,
                    "year": 2021
                },
                "ends_at": {
                    "day": 30,
                    "month": 4,
                    "year": 2025
                },
                "field_of_study": None,
                "degree_name": "BS in data science and application",
                "school": "Indian Institute of Technology, Madras",
                "school_linkedin_profile_url": "https://www.linkedin.com/company/157267/",
                "school_facebook_profile_url": None,
                "description": None,
                "logo_url": "https://s3.us-west-000.backblazeb2.com/proxycurl/company/reachiitm/profile?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0004d7f56a0400b0000000001%2F20241122%2Fus-west-000%2Fs3%2Faws4_request&X-Amz-Date=20241122T034050Z&X-Amz-Expires=1800&X-Amz-SignedHeaders=host&X-Amz-Signature=a89a2886b01f57bbd8f9c3d5eaf2d69dff1d9dbb4b80d4497e1aca10244cf03e",
                "grade": None,
                "activities_and_societies": None
            }
        ],
        "languages": [],
        "languages_and_proficiencies": [],
        "accomplishment_organisations": [],
        "accomplishment_publications": [],
        "accomplishment_honors_awards": [],
        "accomplishment_patents": [],
        "accomplishment_courses": [],
        "accomplishment_projects": [],
        "accomplishment_test_scores": [],
        "volunteer_work": [],
        "certifications": [],
        "connections": 2326,
        "people_also_viewed": [
            {
                "link": "https://www.linkedin.com/in/arshya-moonat?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAADjR1XUBkO6ReGEuOzZN9yzi39l5Lpf7Qto",
                "name": "Arshya Moonat",
                "summary": "BS in Data Science and Applications @IIT Madras | BCA @MCMDAV Chd | Data Science Enthusiast",
                "location": None
            },
            {
                "link": "https://www.linkedin.com/in/yash-choudhary-a348161b6?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAADJPf0gBZTmGtRchW-k9KHosphxGgSeBtyk",
                "name": "Yash Choudhary",
                "summary": "Data analyst @ Homelane | ex - Software engineer(AI) @ Remotasks | IIT Madras 24'",
                "location": None
            },
            {
                "link": "https://www.linkedin.com/in/afrin-gowhar-70b680224?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAADhXE_sBSUEg2ZXXR0q3NxfUR0TsHKrHdHg",
                "name": "Afrin Gowhar",
                "summary": "JOINT SECRETARY, PLACEMENT COUNCIL @IIT MADRAS BS Degree | BS Data Science and Applications @IIT Madras",
                "location": None
            },
            {
                "link": "https://www.linkedin.com/in/deepesh-kumar-dawar-ns7270?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAAD9eTbIBOBwRbuqPKcVeKXZImCrMAM59sA0",
                "name": "Deepesh Kumar Dawar",
                "summary": "Intern @SkyGad | Event Lead @GDSC IIT Madras |Student at IIT Madras",
                "location": None
            },
            {
                "link": "https://www.linkedin.com/in/nivedita-jayaswal?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAAEdDXdYBMi-zqh-p0j2wSU7qhLG8rLuqy90",
                "name": "Nivedita Jayaswal",
                "summary": "Data Engineer | IIT Madras",
                "location": None
            },
            {
                "link": "https://www.linkedin.com/in/pratham-bhalla?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAADgT_RAB5Sm8NKHkTRY6a6gCtC3Kx_lnZgo",
                "name": "Pratham Bhalla",
                "summary": "SIH 2023 Finalist @ISRO | Python | Web Developer | Wordpress | Data Analyst | Educator | Student at IIT Madras",
                "location": None
            },
            {
                "link": "https://www.linkedin.com/in/shivansh-jain-in?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAAC9iFmgBrdYVBg-fW2RRa-bAyTeKGpLtMXM",
                "name": "Shivansh Jain",
                "summary": "AI ML Engineer | LLM Enthusiast | IITM BS Student",
                "location": None
            },
            {
                "link": "https://www.linkedin.com/in/samriddhi-kashyap?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAACSs9IEBw9Rhd80QeNwdB9VBsBmQmbbskU4",
                "name": "Samriddhi Kashyap",
                "summary": "Student at 🎓 IIT Madras | WQU 🇺🇲 Student  | AI & ML Enthusiast",
                "location": None
            },
            {
                "link": "https://www.linkedin.com/in/anubhabpanda?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAABq9aD4Bn4K9-ujLCD_h3YM_p12EPCKykwc",
                "name": "Anubhab Panda",
                "summary": "Senior Data Scientist  | Building AI Products",
                "location": None
            },
            {
                "link": "https://www.linkedin.com/in/ansuj-joshi?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAABpUHCEBHrCHghOv6ZK_dpfcNd0TFaOT1yw",
                "name": "Ansuj Joshi",
                "summary": "Computer Vision @ Cyncly | LTTS | Quantiphi | IIT Roorkee",
                "location": None
            }
        ],
        "recommendations": [],
        "activities": [],
        "similarly_named_profiles": [],
        "articles": [],
        "groups": [
            {
                "profile_pic_url": "https://media.licdn.com/dms/image/v2/D5607AQFxmdPBdDZn1A/group-logo_image-shrink_400x400/group-logo_image-shrink_400x400/0/1664001169054?e=1732852800&v=beta&t=erfLAgvpOh_nibRuatY216XNy3taXovl8co88JMCbyM",
                "name": "Sundarbans House",
                "url": "https://www.linkedin.com/groups/14123539"
            },
            {
                "profile_pic_url": "https://media.licdn.com/dms/image/v2/C4E07AQGgOgwCp6LlnA/group-logo_image-shrink_48x48/group-logo_image-shrink_48x48/0/1651504212399?e=1732852800&v=beta&t=ekuF01nkD-hFkr9vxkZvTiob4Rni4P9lAgSTCrWMBUA",
                "name": "IIT Madras BS Students",
                "url": "https://www.linkedin.com/groups/12653543"
            }
        ],
        "skills": [
            "Research Skills",
            "Vector Databases",
            "TypeScript",
            "AngularJS",
            "Retrieval-Augmented Generation (RAG)",
            "Large Language Models (LLM)",
            "Keras",
            "C (Programming Language)",
            "Artificial Intelligence (AI)",
            "Deep Learning",
            "Data Visualization",
            "Analytical Skills",
            "Java",
            "Data Analysis",
            "Unix",
            "Vue.js",
            "JavaScript",
            "Cascading Style Sheets (CSS)",
            "Business Analytics",
            "Data Science",
            "Scikit-Learn",
            "Machine Learning",
            "SQL",
            "HTML",
            "Pandas",
            "NumPy",
            "Business Data Management",
            "Database Management System (DBMS)",
            "Microsoft Excel",
            "Statistics",
            "Bayesian statistics",
            "Python (Programming Language)"
        ],
        "inferred_salary": None,
        "gender": None,
        "birth_date": None,
        "industry": "Information Technology & Services",
        "extra": None,
        "interests": [],
        "personal_emails": [],
        "personal_numbers": []
    }

# Get all links of employees
def call_proxycurl_company(linkedin_url):
    api_endpoint = "https://nubela.co/proxycurl/api/linkedin/company/employees/"
    api_key = os.getenv("PROXYCURL_API_KEY")
    headers = {'Authorization': 'Bearer ' + api_key}

    params = {
        "country": 'in',
        "enrich_profiles": "skip",
        "page_size": 5,
        "employment_status": "current",
        "sort_by": "recently-joined",
        "resolve_numeric_id": "false",
        "url": linkedin_url
    }
    try:
        response = requests.get(api_endpoint, params=params, headers=headers)
        if response.status_code == 404:
            return None
        else:
            response.raise_for_status()  # Raise an exception for HTTP errors
            api_response = response.json()  # Parse JSON from the response
            return api_response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Get details of employee from proxycurl
def get_employee_profile(linkedin_url):
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    api_key = os.getenv("PROXYCURL_API_KEY")
    headers = {'Authorization': 'Bearer ' + api_key}

    params = {
        'url': linkedin_url,
        'fallback_to_cache': 'on-error',
        'use_cache': 'if-present',
        'skills': 'include',
        'personal_email': 'include',
        'extra': 'include'
    }
    try:
        response = requests.get(api_endpoint, params=params, headers=headers)
        if response.status_code == 404:
            return None
        else:
            response.raise_for_status()  # Raise an exception for HTTP errors
            api_response = response.json()  # Parse JSON from the response
            return api_response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


fields_identifier_agent = Agent(
    role="Fields Identifier",
    goal="Accurately identify different fields required from the given context.",
    verbose=True,
    llm=default_llm,
    allow_delegation=False,
    backstory="""
    You are a skilled Fields Identifier. You have been working as Human Resource Manager for a long time. 
    You have a good understanding of correctly manuvering through the fields required for the given context.
    The context might be a page source or a document or a text or even a json, you can easily identify the fields required.
    You are meticulous and detail-oriented. You dont make any mistakes or mismatch the fields.The fields you need to extract are:
        1. Name 
        2. Linkedin url 
        3. Current job role
        4. Experience in Years
        5. Skills
        6. Location
        7. Education Qualification
""")


confidence_scorer_agent = Agent(
    role="Confidence Scorer",
    goal="Accurately identify if the profile of given person is a match with the given job description and provide the confidence score on the scale of 100.",
    verbose=True,
    llm=default_llm,
    allow_delegation=False,
    backstory="""
        You are a skilled Confidence Scorer with extensive experience as a Human Resource Manager. You possess deep knowledge of various job roles and industries.
        Your excellent people skills enable you to accurately assess whether a person's profile aligns with a given job description.
        You provide a Confidence Score on a scale of 100 based on the match between the profile and the job description. This score is calculated as follows:
            - 50% based on Skills
            - 30% based on Experience
            - 10% based on Designation
            - 10% based on Location
        You are meticulous and detail-oriented, ensuring each assessment is thorough and precise. You need to output the following fields:
            1. Name 
            2. LinkedIn URL 
            3. Current Job Role
            4. Experience in Years
            5. Skills
            6. Location
            7. Education Qualification
            8. Confidence Score
""")

st.title("Resume Analyzer")

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Input for company name
linkedin_url = st.text_input("Enter Linkedin URL Company")

# Submit button
submit_button = st.button("Submit")

# Handle the submit action
if submit_button:
    if uploaded_file is not None and linkedin_url.strip() != "":
        st.write("Step 1: Reading PDF file...")
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.write("Step 2: Parsing JD....")


        results = []

        def get_data(text):
            # contents = get_employees_data()
            # employees_data = contents['employees']
            # for details in employees_data:
            job_description = get_job_description(text)
            st.write("1.",job_description.raw)


            job_description = job_description.raw
            st.write("2.",job_description)

            # job_description = json.dumps(job_description_dict)
            # st.write("3.",job_description)

            st.write("Fetching profile URLs from linkedin")
            contents = call_proxycurl_company(linkedin_url)

            # contents = get_employees_data()
            employees = contents['employees']
            st.write("Starting Each employee processing and Confidence score")
            for employee in employees:
                linkedin_profile = employee['profile_url']
                details = get_employee_profile(linkedin_profile)

                field_identification_task = Task(
                    description=(
                        f"""
                        ```{details}```
    
                        From the context given above, identify the following fields:
                        1. Name 
                        2. Linkedin url 
                        3. Current job role
                        4. Experience in Years
                        5. Skills
                        6. Location
                        7. Latest Education Qualification
    
                        Make sure to follow the given set of rules:
                        1) Perform a detailed analysis of the context and extract the required fields.
                        2) Ensure that no data is missed and the fields are accurately extracted.
                        3) If the **Skills** field is not available, extract or derive it from the **headline**, **about section**, or **experiences section**. For all other fields, if not available, provide the output as "Unavailable".
                        4) Ensure that these are the only fields in the output, as the following process will not work with any other fields.
                        5) Provide the output in JSON format.
    
                        Do not provide any additional text, information, or symbols like ``` in the output, as it will not work with the further process.
                        Make sure to provide the output strictly in JSON format.
                        Important note: Every step mentioned above must be followed to get the required results.
                        Do not include ``` or any other such characters in the output.
                        """
                    ),
                    agent=fields_identifier_agent,
                    expected_output="A json with the values of the required fields",
                )

                confidence_scoring_task = Task(
                    description=(
                        f"""
                        From the job descption, provide the confidence score of that person for the job:
                            Job Description (JD): {job_description}
    
                        Make sure to follow the given set of rules:
                        1) Perform a detailed analysis of the context and identify the required fields for matching the profile with the job description.
                        2) Ensure that no data is missed and the fields are accurately analyzed.
                        3) Provide the confidence score on a scale of 100 based on the match of the profile with the job description. The confidence score should comprise:
                            - 50% for Skills matching the jd
                            - 30% for Total Experience in range as mentioned in jd
                            - 10% for Designation matching
                            - 10% for Location matching
                        4) Use your expertise and knowledge to calculate the confidence score based on the above weightings.
                        5) The output should contain the following fields:
                            1. Name 
                            2. Linkedin URL 
                            3. Current Job Role
                            4. Experience in Years
                            5. Skills
                            6. Location
                            7. Education Qualification
                            8. Confidence Score
                            9. Reasoning
                        6) Ensure that only the above fields are included in the output, as the subsequent process relies on this specific structure.
                        7) Provide the output in JSON format.
    
                        **Important Instructions:**
                        - Do not include any additional text, information, or symbols such as ``` in the output, as it will interfere with the further processing.
                        - Adhere strictly to the JSON format for the output.
                        - Every step mentioned above must be followed precisely to achieve the desired results.
                        - In the `Reasoning` field, provide a clear explanation of why the confidence score was assigned based on the analysis of skills, experience, designation, and/or location.
                        """
                    ),
                    agent=confidence_scorer_agent,
                    context=[field_identification_task],
                    expected_output="A json with the values of the required fields"
                )
                try:
                    crew = Crew(
                        agents=[fields_identifier_agent, confidence_scorer_agent],
                        tasks=[field_identification_task, confidence_scoring_task],
                        verbose=True,
                        allow_delegation=False,
                        cache=True
                    )
                    result = crew.kickoff()
                    results.append(result)
                except Exception as e:
                    st.error(f"Exception when processing the data: {e}")
                    return {'error': str(e)}


        get_data(text)
        # Process the results
        parsed_data = [json.loads(record.raw) for record in results]
        df = pd.json_normalize(parsed_data)

        sort_df = df.sort_values(by='Confidence Score', ascending=False).reset_index(drop=True)
        st.dataframe(sort_df)
        # Save to Excel
        df.to_excel("results.xlsx", index=False)
        sort_df.to_excel("sorted_results.xlsx", index=False)
        st.success("Results saved to results.xlsx")
else:
    st.write("Please upload a PDF file and enter the company name.")




