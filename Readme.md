# Custom Question and Answer Streamlit App

This Streamlit application demonstrates the capabilities of Pre Trained LLM using quora data for question answer system.

## Installation

Follow these steps to install and run the application:

1. Clone the repository

        git clone https://github.com/your-repo-url/question_answer_llm.git
        cd question_answer_llm

2. Create a virtual environment and install dependencies

        python -m venv venv
        source venv/bin/activate # For Windows, use 'venv\Scripts\activate'
        pip install -r requirements.txt


## Usage
### Local Development

Run the Streamlit app with the following command:

    streamlit run app.py

Open your web browser and navigate to http://localhost:8501 to access the application.

### Docker Deployment

Build the Docker image:

    docker build -t question_answer_llm .

Run the Docker image:

    docker run -p 8501:8501 question_answer_llm
