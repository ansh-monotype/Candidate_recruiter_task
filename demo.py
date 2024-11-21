from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import time
import json
import requests
import pandas as pd

load_dotenv()
# Extract vacacny, Skillsets, experince, education qulifaction
def get_job_description():
    return '''
    AI Engineer I / AI Engineer II
    Are you our “Type”?
    Monotype (Global)
    Named "One of the Most Innovative Companies in Design" by Fast Company, Monotype brings brands to
    life through type and technology that consumers engage with every day.
    The company's rich legacy includes a library that can be traced back hundreds of years, featuring famed
    typefaces like Helvetica, Futura, Times New Roman and more.
    Monotype also provides a first-of-its-kind service that makes fonts more accessible for creative
    professionals to discover, license, and use in our increasingly digital world. We work with the biggest global
    brands, and with individual creatives, offering a wide set of solutions that make it easier for them to do
    what they do best: design beautiful brand experiences.
    Monotype Solutions India
    Monotype Solutions India is a strategic center of excellence for Monotype and is a certified Great Place to
    Work® three years in a row. The focus of this fast-growing center spans Product Development, Product
    Management, Experience Design, User Research, Market Intelligence, Research in areas of Artificial
    Intelligence and Machine learning, Innovation, Customer Success, Enterprise Business Solutions, and Sales.
    Headquartered in the Boston area of the United States and with offices across 4 continents, Monotype is
    the world’s leading company in fonts. It’s a trusted partner to the world’s top brands and was named “One
    of the Most Innovative Companies in Design” by Fast Company.
    Monotype brings brands to life through the type and technology that consumers engage with every day.
    The company's rich legacy includes a library that can be traced back hundreds of years, featuring famed
    typefaces like Helvetica, Futura, Times New Roman, and more. Monotype also provides a first-of-its-kind
    service that makes fonts more accessible for creative professionals to discover, license, and use in our
    increasingly digital world. We work with the biggest global brands, and with individual creatives, offering
    a wide set of solutions that make it easier for them to do what they do best: design beautiful brand
    experiences.
    What you will be doing:
    • Design, development and deployment of AI/ML solutions and modelsthat solve complex business
    problems.
    • Understand business requirements, collect, clean, prepare data for machine learning and
    translate them into AI/ML solutions.
    • Implement best practices and come up with innovative ideas.
    • Conduct in-depth research, stay current with the latest advancements in AI/ML, and apply new
    findings to solve complex problems.
    • Develop and maintain scalable, production-ready code for AI models, ensuring high performance
    and reliability.
    • Implement and maintain data pipelines, preprocess data, and ensure data quality for machine
    learning tasks.
    • Work on model optimization and scalability to handle large datasets and high throughput
    requirements.
    What we are looking for:
    • Bachelor's or master's degree in computer science, Machine Learning, or a related field.
    • 0-4 years of hands-on experience in AI and ML engineering.
    • Strong understanding of machine learning concepts, algorithms, and techniques.
    • Proficiency with Prompt engineering, custom chat bots and generative image models like
    Midjourney, DALLE & Stable Diffusion.
    • Proficiency in deep learning frameworks (e.g., TensorFlow, PyTorch) and experience with model
    training and deployment.
    • Proficiency with programming languages (Python, R).
    • Familiarity with data wrangling and manipulation tools (pandas, NumPy).
    • Excellent problem-solving skills and the ability to work on open-ended, ambiguous problems.
    • Strong communication skills, with the ability to convey technical concepts to a non-technical
    audience.
    • Experience with cloud platforms such as AWS, Azure, or GCP.
    Preferred Skills:
    • Strong understanding of computer vision, natural language processing, and other AI-related
    domains.
    • Published research papers or contributions to the AI/ML community is a significant advantage.
    Monotype is an Equal Opportunities Employer. Qualified applicants will receive consideration for employment without regard to race, color, religion, sex, national
    origin, sexual orientation, gender identity, disability or protected veteran status.
'''

def get_employees_data():
    return{
        "employees":[
            {
            "profile_url": "https://www.linkedin.com/in/christian-zizza-a6a00226",
            "profile": {
                "public_identifier": "christian-zizza-a6a00226",
                "profile_pic_url": "https://s3.us-west-000.backblazeb2.com/proxycurl/person/christian-zizza-a6a00226/profile?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0004d7f56a0400b0000000001%2F20241119%2Fus-west-000%2Fs3%2Faws4_request&X-Amz-Date=20241119T074326Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=ab8d6be5f0f277d07afee1f48de7b0799a99f540c13c83830238c548f1f7ee2d",
                "background_cover_image_url": "https://s3.us-west-000.backblazeb2.com/proxycurl/person/christian-zizza-a6a00226/cover?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0004d7f56a0400b0000000001%2F20241119%2Fus-west-000%2Fs3%2Faws4_request&X-Amz-Date=20241119T074326Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=732c3012018b4ad87e908c1d920ef16844ccaa34ed03b14ca246ab0bfbbe4d59",
                "first_name": "Christian",
                "last_name": "Zizza",
                "full_name": "Christian Zizza",
                "follower_count": 1396,
                "occupation": "Enterprise Account Executive at Monotype",
                "headline": "Enterprise Sales @ Monotype",
                "summary": "Results-driven sales professional with a proven track record spanning 13+ years in strategic account management and enterprise software sales. Specializing in C-level and consultative selling, I excel in building and maintaining strong client relationships across various industries, particularly in FinTech, Media, and Financial Services verticals. My expertise spans territory management, customer success, and channel partnerships, consistently delivering outstanding results:\n\n- Recognized as a top performer, earning accolades such as IBM's Best of IBM Award and multiple 100% Club memberships\n- Demonstrated ability to manage and grow multi-million dollar books of business\n- Skilled in leveraging CRM and sales tools to drive business development and strategic account planning\n\nWith a background in offerings covering Advanced Analytics and AI solutions, SaaS, and Digital Quality; I bring a comprehensive understanding of cutting-edge technologies to drive value for clients. My approach combines strong collaboration with a deep commitment to customer success, resulting in high client retention rates and consistent revenue growth. Whether working with enterprise accounts or managing channel partnerships, I'm dedicated to delivering exceptional results and fostering productive business relationships.",
                "country": "US",
                "country_full_name": "United States",
                "city": "Winthrop",
                "state": "Massachusetts",
                "experiences": [
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 10,
                            "year": 2024
                        },
                        "ends_at": None,
                        "company": "Monotype",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/monotype",
                        "company_facebook_profile_url": None,
                        "title": "Enterprise Account Executive",
                        "description": None,
                        "location": "Woburn, Massachusetts, United States",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4D0BAQGAT4pdLOryOA/company-logo_400_400/company-logo_400_400/0/1630534321392/monotype_logo?e=1740009600&v=beta&t=LjPSP5mMaoh18CUyu11oCFChv0uoLXRGC86ynp22Ado"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 9,
                            "year": 2023
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 7,
                            "year": 2024
                        },
                        "company": "Applause",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/applause",
                        "company_facebook_profile_url": None,
                        "title": "Strategic Account Manager & Client Director - FinTech & Media",
                        "description": None,
                        "location": "Boston, Massachusetts, United States",
                        "logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQF0S2U3dVt8lQ/company-logo_400_400/company-logo_400_400/0/1688563679615/applause_logo?e=1740009600&v=beta&t=6jKBMI35csF5LpFi70X_SK3WlvL-K781x0hdZSPCdHk"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 10,
                            "year": 2022
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 5,
                            "year": 2023
                        },
                        "company": "Vendr",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/vendr-co",
                        "company_facebook_profile_url": None,
                        "title": "Enterprise Account Executive",
                        "description": None,
                        "location": "Boston, Massachusetts, United States",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQF7E40yPIN4bg/company-logo_400_400/company-logo_400_400/0/1660053951871/vendr_co_logo?e=1740009600&v=beta&t=I-X--2Rww6sqNerzaUWQWQY3CDu--xuqT5HJ9aGr-JU"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 7,
                            "year": 2021
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 8,
                            "year": 2022
                        },
                        "company": "Applause",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/applause",
                        "company_facebook_profile_url": None,
                        "title": "Strategic Account Manager & Client Director - Financial Services",
                        "description": None,
                        "location": "Boston, Massachusetts, United States",
                        "logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQF0S2U3dVt8lQ/company-logo_400_400/company-logo_400_400/0/1688563679615/applause_logo?e=1740009600&v=beta&t=6jKBMI35csF5LpFi70X_SK3WlvL-K781x0hdZSPCdHk"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2021
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 7,
                            "year": 2021
                        },
                        "company": "IBM",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/ibm",
                        "company_facebook_profile_url": None,
                        "title": "Data Science & AI Account Manager- Insurance ",
                        "description": None,
                        "location": "Boston, Massachusetts, United States",
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGiz5ecgpCtkA/company-logo_400_400/company-logo_400_400/0/1688684715866/ibm_logo?e=1740009600&v=beta&t=hUpWItKPVrhiajQhsH2WVGWaN7Seo5wsM2QFpsln88A"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 7,
                            "year": 2019
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 1,
                            "year": 2021
                        },
                        "company": "IBM",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/ibm",
                        "company_facebook_profile_url": None,
                        "title": "Digital Account Manager - Data Science & AI - Financial Services ",
                        "description": "Team Lead",
                        "location": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGiz5ecgpCtkA/company-logo_400_400/company-logo_400_400/0/1688684715866/ibm_logo?e=1740009600&v=beta&t=hUpWItKPVrhiajQhsH2WVGWaN7Seo5wsM2QFpsln88A"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2016
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 7,
                            "year": 2019
                        },
                        "company": "IBM",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/ibm",
                        "company_facebook_profile_url": None,
                        "title": "Digital Account Manager - Data Science & AI - New England ",
                        "description": "Team Lead",
                        "location": "Greater Boston Area",
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGiz5ecgpCtkA/company-logo_400_400/company-logo_400_400/0/1688684715866/ibm_logo?e=1740009600&v=beta&t=hUpWItKPVrhiajQhsH2WVGWaN7Seo5wsM2QFpsln88A"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2015
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 8,
                            "year": 2016
                        },
                        "company": "Carbonite",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/carbonite",
                        "company_facebook_profile_url": None,
                        "title": "Channel Account Manager",
                        "description": None,
                        "location": "Greater Boston Area",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQEsGl_hsbXSDg/company-logo_400_400/company-logo_400_400/0/1649733833843/carbonite_logo?e=1740009600&v=beta&t=00tI7ohARNptXNqFqeK2XQ7coeBEtzdla67gccJe_Es"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 10,
                            "year": 2012
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 1,
                            "year": 2015
                        },
                        "company": "IBM",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/ibm",
                        "company_facebook_profile_url": None,
                        "title": "Inside Sales Representative- Business Analytics",
                        "description": None,
                        "location": "Littleton, Ma",
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGiz5ecgpCtkA/company-logo_400_400/company-logo_400_400/0/1688684715866/ibm_logo?e=1740009600&v=beta&t=hUpWItKPVrhiajQhsH2WVGWaN7Seo5wsM2QFpsln88A"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 6,
                            "year": 2011
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 10,
                            "year": 2012
                        },
                        "company": "IBM",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/ibm",
                        "company_facebook_profile_url": None,
                        "title": "Business Development Representative",
                        "description": None,
                        "location": "Westford, Ma",
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGiz5ecgpCtkA/company-logo_400_400/company-logo_400_400/0/1688684715866/ibm_logo?e=1740009600&v=beta&t=hUpWItKPVrhiajQhsH2WVGWaN7Seo5wsM2QFpsln88A"
                    }
                ],
                "education": [
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2007
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 12,
                            "year": 2011
                        },
                        "field_of_study": "Minor in Marketing",
                        "degree_name": "Bachelor of Business Administration (B.B.A.)",
                        "school": "Saint Michael's College",
                        "school_linkedin_profile_url": "https://www.linkedin.com/company/25877/",
                        "school_facebook_profile_url": None,
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQFg6hHzLk1kHw/company-logo_400_400/company-logo_400_400/0/1711451044949/saint_michaels_college_logo?e=1740009600&v=beta&t=H0lY2HMlVAS9IITXCPBRf4LG-9n2pMIEOHUm_zJur7s",
                        "grade": None,
                        "activities_and_societies": "Saint Michael's College Men's Ice Hockey & College Librarian"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2003
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 12,
                            "year": 2007
                        },
                        "field_of_study": None,
                        "degree_name": None,
                        "school": "Buckingham Browne & Nichols School",
                        "school_linkedin_profile_url": "https://www.linkedin.com/company/361869/",
                        "school_facebook_profile_url": None,
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQFoDYf1e6XlbQ/company-logo_400_400/company-logo_400_400/0/1630595891335/buckingham_browne__nichols_logo?e=1740009600&v=beta&t=-bxsIcUpypi1DkyDraN4_aW65gwO9bwuI0euLafVTQ0",
                        "grade": None,
                        "activities_and_societies": None
                    }
                ],
                "languages": [],
                "languages_and_proficiencies": [],
                "accomplishment_organisations": [],
                "accomplishment_publications": [],
                "accomplishment_honors_awards": [
                    {
                        "title": "Top ARR Attainment - Enterprise",
                        "issuer": "Vendr",
                        "issued_on": {
                            "day": 1,
                            "month": 3,
                            "year": 2023
                        },
                        "description": "Top 3 in Q1 Net New ARR"
                    },
                    {
                        "title": "2021 Presidents Club",
                        "issuer": "Applause",
                        "issued_on": {
                            "day": 1,
                            "month": 12,
                            "year": 2021
                        },
                        "description": None
                    },
                    {
                        "title": "2020 100% Club",
                        "issuer": "IBM",
                        "issued_on": {
                            "day": 1,
                            "month": 1,
                            "year": 2020
                        },
                        "description": "2020 Q1 Financial Services Rep Of The Quarter"
                    },
                    {
                        "title": "Best of IBM - Top IBM Sellers WW",
                        "issuer": "IBM",
                        "issued_on": {
                            "day": 1,
                            "month": 1,
                            "year": 2020
                        },
                        "description": None
                    },
                    {
                        "title": "2019 100% Club",
                        "issuer": "IBM",
                        "issued_on": {
                            "day": 1,
                            "month": 12,
                            "year": 2019
                        },
                        "description": "2019 Q4 Financial Services Rep Of The Quarter"
                    },
                    {
                        "title": "2018 100% Club",
                        "issuer": "IBM",
                        "issued_on": {
                            "day": 1,
                            "month": 1,
                            "year": 2018
                        },
                        "description": None
                    },
                    {
                        "title": "2017 100% Club",
                        "issuer": "IBM",
                        "issued_on": {
                            "day": 1,
                            "month": 1,
                            "year": 2017
                        },
                        "description": "Named Team Lead For DS & AI Team"
                    },
                    {
                        "title": "#1 Channel Account Manager in New Business Attainment",
                        "issuer": "Carbonite",
                        "issued_on": {
                            "day": 1,
                            "month": 1,
                            "year": 2015
                        },
                        "description": None
                    },
                    {
                        "title": "2013 100% Club",
                        "issuer": "IBM",
                        "issued_on": {
                            "day": 1,
                            "month": 4,
                            "year": 2013
                        },
                        "description": None
                    }
                ],
                "accomplishment_patents": [],
                "accomplishment_courses": [],
                "accomplishment_projects": [],
                "accomplishment_test_scores": [],
                "volunteer_work": [],
                "certifications": [
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2023
                        },
                        "ends_at": None,
                        "name": "Salesloft Platform Export",
                        "license_number": None,
                        "display_source": None,
                        "authority": "Salesloft",
                        "url": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 2,
                            "year": 2012
                        },
                        "ends_at": None,
                        "name": "IBM Global Sales School Graduate",
                        "license_number": None,
                        "display_source": None,
                        "authority": "IBM",
                        "url": None
                    }
                ],
                "connections": 1407,
                "people_also_viewed": [
                    {
                        "link": "https://www.linkedin.com/in/jameshbinder",
                        "name": "James Binder",
                        "summary": "Strategic Account Executive, Tableau at Salesforce",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/kevin-harrigan-5377732a",
                        "name": "Kevin Harrigan",
                        "summary": "Senior Account Manager at BitSight",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/andrewloveland",
                        "name": "Andrew Loveland",
                        "summary": "Strategic Account Executive at Salesforce",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/adriangarcia4",
                        "name": "Adrian Garcia",
                        "summary": "Helping organizations build better quality digital experiences faster and release with confidence",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/yanadulude",
                        "name": "Yana Dulude",
                        "summary": "Senior Account Manager at Unioncrate",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/renee-freeman-5aa181",
                        "name": "Renee Freeman",
                        "summary": "Strategic Solution Delivery Manager @ Applause | Specializing in Financial Services and Customer Experience",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/jmyersdbp",
                        "name": "Jake Myers",
                        "summary": "Account Manager",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/nick-sampson-4926ab41",
                        "name": "Nick Sampson",
                        "summary": "Account Manager Team Lead at Unitrends",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/justin-clermont-252a3713",
                        "name": "Justin Clermont",
                        "summary": "Sr. Account Manager @ Qlik | Driving Sales Growth",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/lane-bachicha-74029264",
                        "name": "Lane Bachicha",
                        "summary": "Quality Assurance Analyst at Applause",
                        "location": None
                    }
                ],
                "recommendations": [
                    "Jack Cruickshank\n\n\n\nChristian is one of the strongest salespeople that I have worked with. He has all of the intangibles of a great salesperson: organized, highly-motivated, charismatic, and a willingness to listen and learn the pains of his customers. I was consistently impressed with his ability build rapport with prospective customers as well as his own colleagues. He was truly a pleasure to work with and any organization is lucky to have him.",
                    "John T.\n\n\n\nChristian is an experienced and expert sales professional who is adept at aligning the needs of his customers with the value that his solutuions provide. He would be an asset for any sales organization and I recommend him highly."
                ],
                "activities": [],
                "similarly_named_profiles": [
                    {
                        "name": "Christian Zizza",
                        "link": "https://au.linkedin.com/in/christian-zizza-5bb09a169",
                        "summary": "Owner of CZ fitness",
                        "location": "Greater Sydney Area"
                    }
                ],
                "articles": [],
                "groups": [
                    {
                        "profile_pic_url": "https://media.licdn.com/dms/image/v2/D4E07AQGq-9dUq8D7GQ/group-logo_image-shrink_400x400/group-logo_image-shrink_400x400/0/1664386507210?e=1732568400&v=beta&t=QWHGG_ojrQ98NxOtNqb0mXcqOqW97JAq8e78fZqKKXA",
                        "name": "Designers Talk: Graphic, web, digital design and creative professionals group",
                        "url": "https://www.linkedin.com/groups/92232"
                    },
                    {
                        "profile_pic_url": "https://media.licdn.com/dms/image/v2/C4D07AQG37VB8M0OUdA/group-logo_image-shrink_48x48/group-logo_image-shrink_48x48/0/1631366309621?e=1732568400&v=beta&t=BqFBGsR4tDXMTJTKrHbFB2k5Rs4XmaeoSPEzE_85muY",
                        "name": "Cloud Computing, SaaS, Data Centre & Virtualization",
                        "url": "https://www.linkedin.com/groups/45151"
                    }
                ],
                "skills": []
            },
            "last_updated": "2024-11-18T20:29:09Z"
        },
        {
            "profile_url": "https://www.linkedin.com/in/kelci-elliott-28608211a",
            "profile": {
                "public_identifier": "kelci-elliott-28608211a",
                "profile_pic_url": "https://s3.us-west-000.backblazeb2.com/proxycurl/person/kelci-elliott-28608211a/profile?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0004d7f56a0400b0000000001%2F20241119%2Fus-west-000%2Fs3%2Faws4_request&X-Amz-Date=20241119T074326Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=03c70d862676d959f3839cc4dcf4f13fde24e460e9347c2562204a6d35cf28f2",
                "background_cover_image_url": "https://s3.us-west-000.backblazeb2.com/proxycurl/person/kelci-elliott-28608211a/cover?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0004d7f56a0400b0000000001%2F20241119%2Fus-west-000%2Fs3%2Faws4_request&X-Amz-Date=20241119T074326Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=e9a87239063bce4e51aa6558b3d1f8994f47a4636b573314d47768ed677fcca4",
                "first_name": "Kelci",
                "last_name": "Elliott",
                "full_name": "Kelci Elliott",
                "follower_count": 1038,
                "occupation": "Foundry Relations and Marketing Manager at Monotype",
                "headline": "Client Relations, E-Commerce Strategy, and Client Marketing Management at Monotype",
                "summary": "I love marrying the practice of marketing strategy with creative marketing ideas. Nothing sets me on fire more than a good marketing campaign that shares a story or inspires action. Im able to strategize and execute digital programs and campaigns across multiple digital channels including email, website, social media, and virtual events. \n\nI am a big dreamer, goal setter, and relationship builder by day. By night, I am a Peloton rider, book lover, and avid community leader. Let’s chat!",
                "country": "US",
                "country_full_name": "United States of America",
                "city": "Louisville",
                "state": "Kentucky",
                "experiences": [
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 9,
                            "year": 2024
                        },
                        "ends_at": None,
                        "company": "Monotype",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/monotype/",
                        "company_facebook_profile_url": None,
                        "title": "Foundry Relations and Marketing Manager",
                        "description": None,
                        "location": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4D0BAQGAT4pdLOryOA/company-logo_400_400/company-logo_400_400/0/1630534321392/monotype_logo?e=1733961600&v=beta&t=LyAml_F14JFA-gO0yVv96NqztpB6wV_mR91lEdzTCI8"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 11,
                            "year": 2021
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 9,
                            "year": 2024
                        },
                        "company": "Monotype",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/monotype/",
                        "company_facebook_profile_url": None,
                        "title": "Marketing Promotions Manager",
                        "description": None,
                        "location": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4D0BAQGAT4pdLOryOA/company-logo_400_400/company-logo_400_400/0/1630534321392/monotype_logo?e=1733961600&v=beta&t=LyAml_F14JFA-gO0yVv96NqztpB6wV_mR91lEdzTCI8"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2021
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 11,
                            "year": 2021
                        },
                        "company": "Southeast Christian Church",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/southeast-christian-church/",
                        "company_facebook_profile_url": None,
                        "title": "Marketing Manager",
                        "description": None,
                        "location": "Louisville, Kentucky, United States",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C560BAQE3SVIZewUmEg/company-logo_400_400/company-logo_400_400/0/1631314447309?e=1733961600&v=beta&t=11NeZxpj2lUVS-MK22n_dYvx3cpwM2irorxUpCHwXfc"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 7,
                            "year": 2019
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 5,
                            "year": 2021
                        },
                        "company": "Southeast Christian Church",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/southeast-christian-church/",
                        "company_facebook_profile_url": None,
                        "title": "Marketing Strategist",
                        "description": None,
                        "location": "Louisville, Kentucky, United States",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C560BAQE3SVIZewUmEg/company-logo_400_400/company-logo_400_400/0/1631314447309?e=1733961600&v=beta&t=11NeZxpj2lUVS-MK22n_dYvx3cpwM2irorxUpCHwXfc"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 4,
                            "year": 2018
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 7,
                            "year": 2019
                        },
                        "company": "Kentucky Department of Juvenile Justice",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/kentucky-department-of-juvenile-justice/",
                        "company_facebook_profile_url": None,
                        "title": "Director of Communications",
                        "description": None,
                        "location": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQGJT_SBWamNSQ/company-logo_400_400/company-logo_400_400/0/1666273521882/kentucky_department_of_juvenile_justice_logo?e=1733961600&v=beta&t=kLGS9xl1rYGWNHPCT47ST7JCxkX3H5Sh4PA21L5UcVo"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2017
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 4,
                            "year": 2018
                        },
                        "company": "KY Justice and Public Safety Cabinet",
                        "company_linkedin_profile_url": None,
                        "company_facebook_profile_url": None,
                        "title": "Assistant Director Of Communications",
                        "description": None,
                        "location": None,
                        "logo_url": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 6,
                            "year": 2017
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 9,
                            "year": 2017
                        },
                        "company": "Kentucky House GOP",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/kentucky-house-gop/",
                        "company_facebook_profile_url": None,
                        "title": "Legislative Assistant",
                        "description": "Reason for leaving: received promotion to a Cabinet-level position.",
                        "location": "Frankfort, Kentucky",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQGdGKr8kbwlRw/company-logo_400_400/company-logo_400_400/0/1630576235889?e=1733961600&v=beta&t=64G58slLlImtZE9hB-ZnLqpu9SwqCchYIk9CFpj6yxc"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 9,
                            "year": 2015
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 6,
                            "year": 2017
                        },
                        "company": "Red Mile Gaming & Racing",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/red-mile-gaming-racing/",
                        "company_facebook_profile_url": None,
                        "title": "Marketing and Promotions Specialist",
                        "description": None,
                        "location": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQHDQZItKIVrRg/company-logo_400_400/company-logo_400_400/0/1675102968941?e=1733961600&v=beta&t=kECPu1E_PhtLeHiJRRVQ19-QS1rduibh0BXPQLsAzAc"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 7,
                            "year": 2014
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 6,
                            "year": 2017
                        },
                        "company": "5-hour ENERGY®",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/5-hour-energy/",
                        "company_facebook_profile_url": None,
                        "title": "Marketing And Promotions Intern",
                        "description": None,
                        "location": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQGHf4EPG8BImg/company-logo_400_400/company-logo_400_400/0/1631348780112?e=1733961600&v=beta&t=vMjNTCs-3k6I3zOMzEa5hoWgOKSKepmrcPhjLIWZeM8"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2016
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 5,
                            "year": 2017
                        },
                        "company": "University of Kentucky Student Government Association",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/university-of-kentucky-student-government-association/",
                        "company_facebook_profile_url": None,
                        "title": "Director of Marketing",
                        "description": None,
                        "location": "University of Kentucky",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C560BAQHCJD13oLLfaQ/company-logo_400_400/company-logo_400_400/0/1630653819779/university_of_kentucky_student_government_association_logo?e=1733961600&v=beta&t=mm6GuY4CRmGC0bg1xpl7DRUNPI0plWad2He8XaITNMI"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2014
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 9,
                            "year": 2016
                        },
                        "company": "University of Kentucky",
                        "company_linkedin_profile_url": "https://www.linkedin.com/school/universityofkentucky/",
                        "company_facebook_profile_url": None,
                        "title": "Marketing & Communications Team Leader",
                        "description": None,
                        "location": "Christian Student Fellowship",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C560BAQEKue1bZKdg6g/company-logo_400_400/company-logo_400_400/0/1656505619159/universityofkentucky_logo?e=1733961600&v=beta&t=zgM_c4eOaOecvodrDW3EKECEvdX6bck0dt0WQy_50fI"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2014
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 5,
                            "year": 2016
                        },
                        "company": "University of Kentucky",
                        "company_linkedin_profile_url": "https://www.linkedin.com/school/universityofkentucky/",
                        "company_facebook_profile_url": None,
                        "title": "Communications Committee Chair",
                        "description": None,
                        "location": "Student Wellness Ambassadors ",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C560BAQEKue1bZKdg6g/company-logo_400_400/company-logo_400_400/0/1656505619159/universityofkentucky_logo?e=1733961600&v=beta&t=zgM_c4eOaOecvodrDW3EKECEvdX6bck0dt0WQy_50fI"
                    }
                ],
                "education": [
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2018
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 1,
                            "year": 2019
                        },
                        "field_of_study": "Business Communications",
                        "degree_name": "Master's Degree",
                        "school": "Mississippi College",
                        "school_linkedin_profile_url": "https://www.linkedin.com/school/mississippi-college/",
                        "school_facebook_profile_url": None,
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQHUl1zBR25EAA/company-logo_400_400/company-logo_400_400/0/1666992796497/mississippi_college_logo?e=1733961600&v=beta&t=SSgsn88bK2fzKNzD3MTTiL4ArpkGFBeHLGO3qNcK0pA",
                        "grade": "4.0",
                        "activities_and_societies": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2013
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 1,
                            "year": 2017
                        },
                        "field_of_study": "Political Science and History",
                        "degree_name": "Bachelor’s Degree",
                        "school": "University of Kentucky",
                        "school_linkedin_profile_url": "https://www.linkedin.com/school/universityofkentucky/",
                        "school_facebook_profile_url": None,
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C560BAQEKue1bZKdg6g/company-logo_400_400/company-logo_400_400/0/1656505619159/universityofkentucky_logo?e=1733961600&v=beta&t=zgM_c4eOaOecvodrDW3EKECEvdX6bck0dt0WQy_50fI",
                        "grade": "3.5",
                        "activities_and_societies": None
                    }
                ],
                "languages": [],
                "languages_and_proficiencies": [],
                "accomplishment_organisations": [
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2019
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2022
                        },
                        "org_name": "Brooklawn Bellwood Ambassador Council",
                        "title": "Board Vice Chair",
                        "description": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2020
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 7,
                            "year": 2021
                        },
                        "org_name": "Junior League of Louisville",
                        "title": "VP Fund Development",
                        "description": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2020
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 7,
                            "year": 2021
                        },
                        "org_name": "Young Professional Association ",
                        "title": "Director of Community Outreach",
                        "description": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2019
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 7,
                            "year": 2020
                        },
                        "org_name": "Young Professionals Association of Louisville",
                        "title": "Director of Diversity and Inclusion",
                        "description": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2018
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 7,
                            "year": 2019
                        },
                        "org_name": "Junior League of Louisville",
                        "title": "Social Media Committee Chair",
                        "description": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2016
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2017
                        },
                        "org_name": "University of Kentucky Student Government Association",
                        "title": "Director of Marketing",
                        "description": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2014
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 3,
                            "year": 2017
                        },
                        "org_name": "Leadership Exchange Ambassadors",
                        "title": "Leadership team and Marketing committee",
                        "description": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2014
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 10,
                            "year": 2016
                        },
                        "org_name": "Christian Student Fellowship",
                        "title": "Communications Lead",
                        "description": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 3,
                            "year": 2015
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 3,
                            "year": 2016
                        },
                        "org_name": "NASPA - Student Affairs in Higher Education",
                        "title": "Student Wellness Area Director",
                        "description": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 4,
                            "year": 2014
                        },
                        "ends_at": {
                            "day": 1,
                            "month": 9,
                            "year": 2015
                        },
                        "org_name": "Office of Student Wellness at the University of Kentucky",
                        "title": "Executive Director",
                        "description": None
                    }
                ],
                "accomplishment_publications": [],
                "accomplishment_honors_awards": [
                    {
                        "title": "Spirit of the League",
                        "issuer": "Junior League of Louisville",
                        "issued_on": {
                            "day": 1,
                            "month": 1,
                            "year": 2019
                        },
                        "description": None
                    },
                    {
                        "title": "Rising Leader",
                        "issuer": "Kentucky Employees Charitable Campaign",
                        "issued_on": {
                            "day": 1,
                            "month": 12,
                            "year": 2018
                        },
                        "description": None
                    },
                    {
                        "title": "Kentucky Colonel",
                        "issuer": "Governor Matt Bevin",
                        "issued_on": {
                            "day": 1,
                            "month": 11,
                            "year": 2017
                        },
                        "description": "The commission of Kentucky Colonel is the highest title of honor bestowed by the Governor of Kentucky. It is recognition of an individual’s noteworthy accomplishments and outstanding service to our community, state and nation."
                    },
                    {
                        "title": "William C. Parker Scholar",
                        "issuer": None,
                        "issued_on": {
                            "day": 1,
                            "month": 8,
                            "year": 2013
                        },
                        "description": None
                    },
                    {
                        "title": "Foster J. Sanders Leadership Scholar",
                        "issuer": "Louisville Male Traditional High School",
                        "issued_on": None,
                        "description": None
                    }
                ],
                "accomplishment_patents": [],
                "accomplishment_courses": [],
                "accomplishment_projects": [],
                "accomplishment_test_scores": [],
                "volunteer_work": [
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 4,
                            "year": 2024
                        },
                        "ends_at": None,
                        "title": "Diversity, Equity, and Inclusion Committee Member",
                        "cause": "HUMAN_RIGHTS",
                        "company": "Monotype",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/monotype/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4D0BAQGAT4pdLOryOA/company-logo_400_400/company-logo_400_400/0/1630534321392/monotype_logo?e=1733961600&v=beta&t=LyAml_F14JFA-gO0yVv96NqztpB6wV_mR91lEdzTCI8"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 6,
                            "year": 2020
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 5,
                            "year": 2021
                        },
                        "title": "Director Of Community Outreach",
                        "cause": None,
                        "company": "Young Professionals Association of Louisville (YPAL)",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/ypal/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQFGRNTXVOCq7w/company-logo_400_400/company-logo_400_400/0/1631305491762?e=1733961600&v=beta&t=yrlqWOE7JXsgABvV7k5_6xIwQbh96McWkl7hxJFhl4Q"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2019
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 6,
                            "year": 2020
                        },
                        "title": "Director of Diversity and Inclusion",
                        "cause": None,
                        "company": "Young Professionals Association of Louisville (YPAL)",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/ypal/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQFGRNTXVOCq7w/company-logo_400_400/company-logo_400_400/0/1631305491762?e=1733961600&v=beta&t=yrlqWOE7JXsgABvV7k5_6xIwQbh96McWkl7hxJFhl4Q"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 6,
                            "year": 2020
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 6,
                            "year": 2021
                        },
                        "title": "Vice President Of Fund Development",
                        "cause": None,
                        "company": "Junior League of Louisville",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/juniorleaguelouisville/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQG3ExiHWhvrGQ/company-logo_400_400/company-logo_400_400/0/1725473612885?e=1733961600&v=beta&t=1w3CtunxrpKrw_yJUWx5T186M5dKH7KxP41BxpAQGvI"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 6,
                            "year": 2019
                        },
                        "ends_at": {
                            "day": 30,
                            "month": 4,
                            "year": 2020
                        },
                        "title": "Social Media Committee Chair",
                        "cause": None,
                        "company": "Junior League of Louisville",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/juniorleaguelouisville/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQG3ExiHWhvrGQ/company-logo_400_400/company-logo_400_400/0/1725473612885?e=1733961600&v=beta&t=1w3CtunxrpKrw_yJUWx5T186M5dKH7KxP41BxpAQGvI"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2018
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 8,
                            "year": 2019
                        },
                        "title": "Communications Council Member",
                        "cause": None,
                        "company": "Junior League of Louisville",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/juniorleaguelouisville/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQG3ExiHWhvrGQ/company-logo_400_400/company-logo_400_400/0/1725473612885?e=1733961600&v=beta&t=1w3CtunxrpKrw_yJUWx5T186M5dKH7KxP41BxpAQGvI"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2020
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 7,
                            "year": 2022
                        },
                        "title": "Member Of The Board Of Advisors",
                        "cause": None,
                        "company": "LANE OF ROSES",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/lane-of-roses/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4D0BAQEPDAGz1bKI0A/company-logo_400_400/company-logo_400_400/0/1631366721745?e=1733961600&v=beta&t=T-MugSXcjqnwzYwawV-az8b9j6weTLFBt0D05T1ueD0"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2017
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 8,
                            "year": 2020
                        },
                        "title": "Social Media Assistant",
                        "cause": None,
                        "company": "LANE OF ROSES",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/lane-of-roses/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4D0BAQEPDAGz1bKI0A/company-logo_400_400/company-logo_400_400/0/1631366721745?e=1733961600&v=beta&t=T-MugSXcjqnwzYwawV-az8b9j6weTLFBt0D05T1ueD0"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2019
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 1,
                            "year": 2021
                        },
                        "title": "Vice Chairman Of The Board",
                        "cause": None,
                        "company": "Bellewood & Brooklawn",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/bellewoodandbrooklawn/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQFFGwWqM4GVMQ/company-logo_400_400/company-logo_400_400/0/1724075016843/bellewoodandbrooklawn_logo?e=1733961600&v=beta&t=wJ0P817KtjxNkXgnzVRh0wfqfR920VW0l18SKeCvsv0"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 9,
                            "year": 2020
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 8,
                            "year": 2021
                        },
                        "title": "Board Member",
                        "cause": None,
                        "company": "The Miss America Opportunity",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/missamerica/",
                        "description": "Miss Louisiana / Miss Acadiana",
                        "logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGVsKjGft-A5g/company-logo_400_400/company-logo_400_400/0/1630657017350/miss_america_organization_logo?e=1733961600&v=beta&t=YP76Wi85Ldtos4jOfp5Vp7OxtZsZAxUhp4G0trVcWw8"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 5,
                            "year": 2016
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 7,
                            "year": 2016
                        },
                        "title": "Summer Camp Counselor",
                        "cause": "CHILDREN",
                        "company": "Pine Cove",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/pine-cove/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C560BAQEV5xu_9ehuBg/company-logo_400_400/company-logo_400_400/0/1631309576693?e=1733961600&v=beta&t=EJ29dO_MBp5E4rvjG1jTQRIO6Zf3QF0XFkd1RfLAKEY"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 8,
                            "year": 2014
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 12,
                            "year": 2015
                        },
                        "title": "Student Volunteer",
                        "cause": "HEALTH",
                        "company": "Ronald McDonald House Charities of Kentuckiana",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/rmhckentuckiana/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQG2pKs9BxSOSw/company-logo_400_400/company-logo_400_400/0/1630636666572/rmhckentuckiana_logo?e=1733961600&v=beta&t=Uwbpi5NqAyhG9rKxSX5ObIUFo8cV3GJOpS3OJfFug_A"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2013
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 5,
                            "year": 2015
                        },
                        "title": "Hope Lodge Leader",
                        "cause": "HEALTH",
                        "company": "American Cancer Society",
                        "company_linkedin_profile_url": "https://www.linkedin.com/company/american-cancer-society/",
                        "description": None,
                        "logo_url": "https://media.licdn.com/dms/image/v2/D560BAQG-6ma23zk0bw/company-logo_400_400/company-logo_400_400/0/1664374536326/american_cancer_society_logo?e=1733961600&v=beta&t=31WtL4TzoDHwU_WUmZHdhoKmFNy_sMdF0owzKwIw5Dk"
                    }
                ],
                "certifications": [
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 1,
                            "year": 2024
                        },
                        "ends_at": None,
                        "name": "Free-to-Paid Conversion",
                        "license_number": None,
                        "display_source": "credly.com",
                        "authority": "Appcues",
                        "url": "https://www.credly.com/badges/b7413aa3-688b-4005-b067-e2c8a8d19c71/linked_in_profile"
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 3,
                            "year": 2022
                        },
                        "ends_at": {
                            "day": 31,
                            "month": 3,
                            "year": 2025
                        },
                        "name": "Google Analytics",
                        "license_number": None,
                        "display_source": None,
                        "authority": "Google",
                        "url": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 9,
                            "year": 2021
                        },
                        "ends_at": None,
                        "name": "Ecommerce Marketing",
                        "license_number": None,
                        "display_source": None,
                        "authority": "HubSpot",
                        "url": None
                    },
                    {
                        "starts_at": {
                            "day": 1,
                            "month": 7,
                            "year": 2019
                        },
                        "ends_at": None,
                        "name": "Operations and Strategy",
                        "license_number": None,
                        "display_source": None,
                        "authority": "Public Affairs Council",
                        "url": None
                    }
                ],
                "connections": 500,
                "people_also_viewed": [
                    {
                        "link": "https://www.linkedin.com/in/romeo-ruz-88916214",
                        "name": "Romeo Ruz",
                        "summary": "Senior Manager, Marketing at Monotype",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/hunter-mitchell-010b99159",
                        "name": "Hunter Mitchell",
                        "summary": "Owner and operator of 'Photography by Hunter Drake'",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/kristiandudgeon",
                        "name": "Kristian Dudgeon",
                        "summary": "Multimedia Marketing & Communications Professional",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/kathryn-marino-6358741a",
                        "name": "Kathryn Marino",
                        "summary": "Humana Employer Group Marketing",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/max-hosseini-b7008476",
                        "name": "Max Hosseini",
                        "summary": "Director SEO, Marketing Analytics",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/ian-njiru",
                        "name": "Ian Njiru",
                        "summary": "Content Writer | Virtual Assistant Seeking New Opportunities",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/destinee-farr-4a3230229",
                        "name": "Destinee Farr",
                        "summary": "Creative Director at Trinity Chapel Assembly Of God",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/danieljackman",
                        "name": "Daniel Jackman",
                        "summary": "Results Driven Leader and Collaborator",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/mcpflug",
                        "name": "Mary Catherine Pflug",
                        "summary": "Senior Director, Partners, Inventory & E-Commerce",
                        "location": None
                    },
                    {
                        "link": "https://www.linkedin.com/in/natalie-coruzzi-390608224",
                        "name": "Natalie Coruzzi",
                        "summary": "Sincerity Homes",
                        "location": None
                    }
                ],
                "recommendations": [],
                "activities": [
                    {
                        "title": "I'm a social media manager and my personal Instagram has 109 followers.  When ya'll see someone in social with nothing to show on their personal…",
                        "link": "https://www.linkedin.com/posts/jacobshipley_socialmedia-marketing-activity-7016387766309527553-L8J6",
                        "activity_status": "Liked by Kelci Elliott"
                    },
                    {
                        "title": "If any companies are doing layoffs this week, I think they should probably be permanently removed from the \"Best Companies to Work For List.\"Laying…",
                        "link": "https://www.linkedin.com/posts/carolyn-christie-bb15656_carolyntherecruiter-staffing-sales-activity-7011738519161233409-xfQI",
                        "activity_status": "Liked by Kelci Elliott"
                    },
                    {
                        "title": "Set the boundaries and stop overworking. What are your tips for good work-life balance?s/o Katy Leeson",
                        "link": "https://www.linkedin.com/posts/linkedin_set-the-boundaries-and-stop-overworking-activity-7009094731246776321-gWwG",
                        "activity_status": "Liked by Kelci Elliott"
                    },
                    {
                        "title": "Sweet as sugar to see these big beautiful slabs in Design Week’s ’22 pop tops. (🙌 Lilia Quinaud—kristie malivindi— Jones Knowles Ritchie—Juan…",
                        "link": "https://www.linkedin.com/posts/chasnix_design-weeks-most-popular-news-stories-of-activity-7010678130994479104-HtAW",
                        "activity_status": "Liked by Kelci Elliott"
                    },
                    {
                        "title": "I'm always rambling about typography or type setting and how crucial I think it is and the impact that it can have when used smartly to tell a story…",
                        "link": "https://www.linkedin.com/posts/evan-george-design_monotypes-marie-boulanger-on-the-nostalgic-activity-7009156249225187329-O9L_",
                        "activity_status": "Liked by Kelci Elliott"
                    },
                    {
                        "title": "7 phrases I stopped using in 2022 (and what I replaced them with):1️⃣ \"Does that make sense?\" --> \"What questions can I answer for you?\"2️⃣ \"I…",
                        "link": "https://www.linkedin.com/posts/brianna-doe_careergrowth-professionaldevelopment-communicationskills-activity-7006666760154230784-d1eN",
                        "activity_status": "Liked by Kelci Elliott"
                    },
                    {
                        "title": "Why did I leave working at church for a sales career?After years of back and forth, I made the jump into sales 4 months ago. It has been nothing…",
                        "link": "https://www.linkedin.com/posts/drew-davis-salesleader_purpose-mindset-growth-activity-7006361119137112064-njg9",
                        "activity_status": "Liked by Kelci Elliott"
                    },
                    {
                        "title": "I was interviewed by Julien Fincker about football, my foundry and my way into type design, … It was a fun interview, you should read it ;)…",
                        "link": "https://www.linkedin.com/posts/moritz-kleinsorge-62487b183_get-to-know-moritz-kleinsorge-activity-7006935868187758593-F7mQ",
                        "activity_status": "Liked by Kelci Elliott"
                    },
                    {
                        "title": "On Wednesday I was incredibly honoured to speak at University of Reading Department of Typography & Graphic Communication for their Baseline Shift…",
                        "link": "https://www.linkedin.com/posts/marie-boulanger-59043258_typography-experience-typedesign-activity-7006971217769320448-Zyp7",
                        "activity_status": "Liked by Kelci Elliott"
                    },
                    {
                        "title": "After six months, most of the 33 companies and 903 workers in a worldwide pilot program are unlikely to ever go back to a standard working week. None…",
                        "link": "https://www.linkedin.com/posts/cnn_after-six-months-most-of-the-33-companies-activity-7003752880759349248-Ro1x",
                        "activity_status": "Liked by Kelci Elliott"
                    }
                ],
                "similarly_named_profiles": [],
                "articles": [],
                "groups": [],
                "skills": []
            },
            "last_updated": "2024-09-06T17:41:19Z"
        }
        ]

    }




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


job_description = get_job_description()

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
        3. Experiences
        4. Experience in Years
        5. Skills
        6. Location
""")


confidence_scorer_agent = Agent(
    role="Confidence Scorer",
    goal="Accurately identify if the profile of given person is a match with the given job description and provide the confidence score on the scale of 5.",
    verbose=True,
    llm=default_llm,
    allow_delegation=False,
    backstory="""
    You are a skilled Confidence Scorer. You have been working as Human Resource Manager for a long time. You know a lot about different jobs.
    You have a good people skills and can easily identify if the profile of given person is a match with the given job description.
    You can accurately provide the confidence score on the scale of 5 based on the match of the profile with the job description.
    You are meticulous and detail-oriented. You need to output the given fields:
            1. Name 
            2. Linkedin url 
            3. Experiences
            4. Experience in Years
            5. Skills
            6. Location
            7. Confidence Score
            8. Reasoning
""")

results = []
def get_data():
    contents = get_employees_data()
    employees = contents['employees']
    for details in employees:
        field_identification_task = Task(
            description=(
            f"""
                ```{details}```

                From the context given above, identify the given fields:
                1. Name 
                2. Linkedin url 
                3. Experiences
                4. Last company role
                5. Total Experience in Years
                6. Skills
                7. Latest education
                7. Location
                
                Make sure to follow the given set of rules
                1) Perform a detailed analysis of the context and extract the required fields.
                2) Make sure that no data is missed out and the fields are accurately extracted.
                3) If any field is not available, provide the output as "Unavailable".
                4) Make sure these are the only fields in the output as the following process will not work with any other fields.
                5) Provide the output in the json format.

                Do not provide any other texts or information or symbols like ``` in the output as it will not work with the further process.
                Make sure the provide the output in the json format.
                Important note: Every step mentioned above must be followed to get the required results.
                Do not provide any other texts or information in the output as it will not work with the further process.
                Do not include ``` or any other such characters in the output.
            """
            ),
            agent=fields_identifier_agent,
            expected_output="A json with the values of the required fields",
        )
        confidence_scoring_task = Task(
            description=(
            f"""
                From the job given context, provide the confidence score of that person for the job:
                    {job_description}
                
                Make sure to follow the given set of rules
                1) Perform a detailed analysis of the context and identify the required fields for matching the profile with the job description.
                2) Make sure that no data is missed out and the fields are accurately analysed.
                3) Provide the confidence score on the scale of 5 based on the match of the profile with the job description.
                4) Use your expertise and knowledge to provide the confidence score.
                5) The output should contain the given fields:
                    1. Name 
                    2. Linkedin url 
                    3. Experiences
                    4. Last company role
                    5. Total Experience in Years
                    6. Skills
                    7. Location
                    8. Confidence Score
                    9. Reasoning
                6) Use your knowledge to provide the fields and the confidence score.
                7) Make sure these are the only fields in the output as the following process will not work with any other fields.
                8) Provide the output in the json format.

                Do not provide any other texts or information or symbols like ``` in the output as it will not work with the further process.
                Make sure the provide the output in the json format.
                Important note: Every step mentioned above must be followed to get the required results.
                Do not provide any other texts or information in the output as it will not work with the further process.
                Do not include ``` or any other such characters in the output.
            """
            ),
            agent= confidence_scorer_agent,
            context = [field_identification_task],
            expected_output="A json with the values of the required fields"
        )
        try:
            trip_crew_subsidiary_research = Crew(
                agents=[
                    fields_identifier_agent, confidence_scorer_agent
                ],
                tasks=[
                    field_identification_task, confidence_scoring_task
                ],
                verbose=True,
                allow_delegation=False,
                cache=True
            )
            result = trip_crew_subsidiary_research.kickoff()
            results.append(result)
        except Exception as e:
            print(f"Exception when getting links for company structures for private company: {e}")
            return {'error': str(e)}
get_data()
print(results)
parsed_data = [json.loads(record.raw) for record in results]
df = pd.json_normalize(parsed_data)
df.to_excel("./results.xlsx")


def call_proxycurl_company(linkedin_url):
    api_endpoint = "https://nubela.co/proxycurl/api/linkedin/company/resolve"
    api_key = os.getenv("proxycurl_api_key")
    headers = {"Authorization": "Bearer " + api_key}

    params = {
        "country": 'us',
        "enrich_profiles": "enrich",
        "page_size": 10,
        "employment_status": "current",
        "sort_by": "recently_joined",
        "resolve_numeric_id": False,
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