import os
import nltk
import ssl
import streamlit as st
import random
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL context fix for nltk
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },
    {
        "tag": "online_learning",
        "patterns": ["What are online courses?", "Can you explain e-learning?", "What is an online Bootcamp?"],
        "responses": ["Online courses are educational programs available on the internet that allow learners to study at their own pace.", "E-learning is a method of education that uses electronic resources, often through the internet.", "An online Bootcamp is an intensive training program designed to teach specific skills in a short period of time."],
        "context": [""]
    },
    {
        "tag": "certifications",
        "patterns": ["What are professional certifications?", "Why should I get certified?", "Are certifications useful?"],
        "responses": ["Professional certifications validate your skills and knowledge in a specific field.", "Getting certified can improve your job prospects and credibility in the industry.", "Certifications help demonstrate your expertise and can lead to career growth."],
        "context": [""]
    },
    {
        "tag": "data_science",
        "patterns": ["What is Data Science?", "Can you explain Data Science?", "Tell me about Data Science"],
        "responses": ["Data Science is a field that uses scientific methods, algorithms, and systems to extract insights from data.", "It involves statistics, programming, and machine learning to analyze and interpret data.", "Data Science helps businesses make data-driven decisions."],
        "context": [""]
    },
    {
        "tag": "cyber_security",
        "patterns": ["What is Cyber Security?", "How does Cyber Security work?", "Why is Cyber Security important?"],
        "responses": ["Cyber Security is the practice of protecting systems, networks, and data from digital attacks.", "It involves measures like encryption, firewalls, and monitoring to prevent cyber threats.", "Cyber Security is crucial for protecting personal and business information from hackers."],
        "context": [""]
    },
    {
        "tag": "cloud_computing",
        "patterns": ["What is Cloud Computing?", "Explain Cloud Computing", "How does Cloud Computing work?"],
        "responses": ["Cloud Computing allows users to access and store data over the internet instead of on local servers.", "It enables businesses to use computing resources on demand without owning physical infrastructure.", "Cloud services like AWS, Azure, and Google Cloud offer scalable and flexible computing solutions."],
        "context": [""]
    },
    {
        "tag": "artificial_intelligence",
        "patterns": ["What is Artificial Intelligence?", "Tell me about AI", "How does AI work?"],
        "responses": ["Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.", "AI enables computers to learn, reason, and make decisions without explicit programming.", "AI technologies like machine learning and neural networks are transforming industries worldwide."],
        "context": [""]
    },
    {
        "tag": "programming_languages",
        "patterns": ["What are popular programming languages?", "Which coding language should I learn?", "Tell me about Python"],
        "responses": ["Some popular programming languages include Python, Java, JavaScript, C++, and SQL.", "The best language to learn depends on your goals—Python for data science, JavaScript for web development, etc.", "Python is widely used in AI, web development, and data science due to its simplicity and versatility."],
        "context": [""]
    },
    {
        "tag": "digital_marketing",
        "patterns": ["What is Digital Marketing?", "Can you explain SEO?", "How does Social Media Marketing work?"],
        "responses": ["Digital Marketing involves promoting products or services online using various strategies like SEO and social media marketing.", "SEO (Search Engine Optimization) improves a website’s visibility in search engine results.", "Social Media Marketing uses platforms like Facebook and Instagram to engage with audiences and promote brands."],
        "context": [""]
    },
    {
        "tag": "career_growth",
        "patterns": ["How can I advance my career?", "What skills are in demand?", "How do I get a better job?"],
        "responses": ["Upskilling through certifications and courses can help advance your career.", "In-demand skills include data science, cloud computing, cyber security, and digital marketing.", "Building a strong resume, networking, and continuous learning can improve your job prospects."],
        "context": [""]
    },
    {
        "tag": "free_courses",
        "patterns": ["Are there free courses available?", "Where can I find free learning resources?", "Can I learn new skills for free?"],
        "responses": ["Yes, many platforms offer free courses in various fields.", "You can find free learning resources on platforms like Coursera, Udemy, and Khan Academy.", "Some websites provide free trial courses or open-source learning materials."],
        "context": [""]
    },
    {
        "tag": "admission_process",
        "patterns": ["How to apply for admission?", "Admission process", "I want to join Avanthi", "How can I get admission?"],
        "responses": ["Admissions are open! You can apply online at our official website or visit our campus in Narsipatnam. Admissions usually start in June."]
    },
    {
        "tag": "courses_offered",
        "patterns": ["What courses are available?", "List of departments", "Which B.Tech branches are there?", "Programs offered?"],
        "responses": ["We offer B.Tech courses in CSE, ECE, EEE, Civil, and Mechanical Engineering. We also offer M.Tech and diploma courses."]
    },
    {
        "tag": "fee_structure",
        "patterns": ["What is the fee structure?", "Tell me about fees", "How much is the tuition fee?"],
        "responses": ["The tuition fee for B.Tech is ₹50,000 per year as per APSCHE guidelines. Additional fees apply for hostels and transport."]
    },
    {
        "tag": "placements",
        "patterns": ["Do you provide placements?", "Placement companies?", "Job opportunities?"],
        "responses": ["Yes! We have a placement cell that invites companies like Infosys, TCS, Wipro, and local industries for campus recruitment."]
    },
    {
        "tag": "hostel_facilities",
        "patterns": ["Is there a hostel?", "Do you provide hostel facilities?", "Hostel details?"],
        "responses": ["Yes, we have separate hostel facilities for boys and girls with mess and security. Hostel fees are separate from tuition."]
    },
    {
        "tag": "contact_info",
        "patterns": ["What is the contact number?", "College address?", "How can I reach you?"],
        "responses": ["You can contact us at +91-xxxxxxxxxx or visit our campus at Avanthi Institute of Engineering & Technology, Narsipatnam, Anakapalli District."]
    },
    {
        "tag": "cse_course_info",
        "patterns": ["Tell me about the B.Tech in CSE program", "CSE course details","What is CSE?"],
        "responses": [ "CSE focuses on programming, data structures, algorithms, AI/ML, and software engineering. Duration is 4 years."],
    },
    {
        "tag": "cse_syllabus",
        "patterns": ["Syllabus for CSE","Subjects in CSE", "CSE curriculum"],
        "responses": ["CSE includes Programming in C, Data Structures, Algorithms, DBMS, AI, ML, Cloud Computing, and IoT."],
    },
    {
        "tag": "cse_faculty",
        "patterns": ["CSE faculty","Who teaches in CSE?","CSE department head"],
        "responses": ["CSE faculty includes Prof. U Nanaji (HOD), Prof. Chiranjeevi(CD),Prof. Vara Prasad(Flat), Prof. DhanaLakshmi (Computer Networks, CNS),Prof. Sagar Kumar(C,Python,C++)"],
    },
    {
        "tag": "ece_course_info",
        "patterns": [ "Tell me about the ECE program", "ECE course details", "What is ECE?"],
        "responses": [ "ECE deals with electronics, communication systems, embedded systems, and VLSI. Duration is 4 years."],
    },
    {
        "tag": "ece_syllabus",
        "patterns": ["Syllabus for ECE", "Subjects in ECE","ECE curriculum"],
        "responses": ["ECE includes Digital Electronics, Signals & Systems, Analog Communication, Microprocessors, VLSI, and Embedded Systems."],
    },
    {
        "tag": "ece_faculty",
        "patterns": ["ECE faculty","Who teaches VLSI?","ECE department head" ],
        "responses": [ "ECE faculty includes Prof. Prasad (HOD), Prof. Divya (VLSI), Prof. Kumar (Communication Systems)."],
    },
    {
        "tag": "eee_course_info",
        "patterns": [ "Tell me about the EEE program","EEE course details","What is EEE?"],
        "responses": [ "EEE focuses on electrical machines, power systems, control systems, and renewable energy. Duration is 4 years."],
    },
    {
        "tag": "eee_syllabus",
        "patterns": ["Syllabus for EEE","Subjects in EEE","EEE curriculum"],
        "responses": [ "EEE includes Electrical Machines, Power Electronics, Control Systems, Switchgear, and Renewable Energy."],
    },
    {
        "tag": "eee_faculty",
        "patterns": [ "EEE faculty","Who teaches Control Systems?","EEE department head"],
        "responses": [ "EEE faculty includes Prof. Ramesh (HOD), Prof. Sudha (Power Systems), Prof. Venu (Control Systems)."],
    },
    {
        "tag": "mech_course_info",
        "patterns": ["Tell me about the Mechanical program","Mech course details","What is Mechanical Engineering?"],
        "responses": ["Mechanical Engineering focuses on machine design, thermodynamics, manufacturing, and robotics. Duration is 4 years."],
    },
    {
        "tag": "mech_syllabus",
        "patterns": ["Syllabus for Mechanical", "Subjects in Mech", "Mechanical curriculum"],
        "responses": ["Mechanical Engineering includes Thermodynamics, Fluid Mechanics, Machine Design, CAD/CAM, and Robotics" ],
    },
    {
        "tag": "mech_faculty",
        "patterns": [ "Mechanical faculty","Who teaches Thermodynamics?","Mechanical department head"],
        "responses": ["Mechanical faculty includes Prof. Kishore (HOD), Prof. Praveen (Thermodynamics), Prof. Suman (Machine Design)."],
    },
    {
        "tag": "exam_schedule",
        "patterns": [ "When are the semester exams?", "Exam timetable", "Exam schedule" ],
        "responses": [ "The semester exams start from 10th May 2025. Check the official website or notice board for updated exam timetables."],
    },
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand your speech."
    except sr.RequestError:
        return "Sorry, my speech service is down."

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            speak(response)
            return response

def main():
    st.title("COLLEGE CHATBOT FOR VISITORS/GUESTS")
    st.write("Welcome! Type your message or use the voice input.")

    # Text input
    user_input = st.text_input("You (text):")

    # Voice input button
    if st.button("Use Voice Input"):
        user_input = recognize_speech()
        st.write(f"You (voice): {user_input}")

    # Process input and generate response
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()