import json
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

# Set API Key
os.environ["OPENAI_API_KEY"] = ""

# Literal Types for goals and technologies
GoalType = Literal[
    "AI Voice Assistant", "Substance Detection", "Face Recognition", "Financial Processing",
    "Hate Speech Detection", "Image Tagging", "Automatic Skill Assessment", "Data Grouping",
    "Copyrighted Content Detection", "Recidivism Prediction", "Chatbot", "Scheduling",
    "Image Cropping", "Content Recommendation", "Smart Devices", "License Plate Recognition",
    "Autonomous Drones", "NSFW Content Detection", "Automatic Stock Trading", "Translation",
    "Gunshot Detection", "Market Forecasting", "Audio Localization", "Robotic Manipulation",
    "Activity Tracking", "Deepfake Video Generation", "Content Search", "Behavioral Modeling",
    "Visual Art Generation", "Autonomous Driving", "Code Generation", "Automated Content Curation",
    "Social Media Content Generation", "Question Answering", "Threat Detection"
]

TechnologyType = Literal[
    "Multimodal Learning", "Content-based Filtering", "Diverse Data", "Intermediate modeling",
    "Image Segmentation", "Autoencoder", "Distributional Learning", "Language Modeling",
    "Satellite Imaging", "Image Classification", "Regression", "Face Detection",
    "Classification", "Collaborative Filtering", "3D reconstruction", "Ensemble Aggregation",
    "Optical Character Recognition", "Character NGrams", "Acoustic Fingerprint",
    "Convolutional Neural Network", "Spectrogram", "Neural Network", "Gesture Recognition",
    "Visual Object Detection", "Transformer", "Clustering", "Generative Adversarial Network",
    "Keyword Filtering", "Automatic Speech Recognition", "Geolocation Data",
    "Siamese Network", "Recurrent Neural Network", "Acoustic Triangulation"
]


# Schema definition using Pydantic
class SearchSchema(BaseModel):
    goals: list[GoalType] = Field(description="The list of goals specified in the request")
    technologies: list[TechnologyType] = Field(description="The list of technologies specified in the request")


# Set up Pydantic parser and ChatOpenAI
temperature = 0.0
model = "gpt-4"
llm = ChatOpenAI(model=model, temperature=temperature)
pydantic_parser = PydanticOutputParser(pydantic_object=SearchSchema)
format_instructions = pydantic_parser.get_format_instructions()

# Prompt template
EXTRACTION_PROMPT = """
Your goal is to understand and parse out the user's goal and technology extraction request.

{format_instructions}

Goal and Technology extraction Request:
{request}
"""

prompt = ChatPromptTemplate.from_template(
    template=EXTRACTION_PROMPT,
    partial_variables={
        "format_instructions": format_instructions
    }
)

# Extraction chain setup
extraction_chain = {"request": lambda x: x["request"]} | prompt | llm


def extract_goal_tech(request_text):
    """Invoke the extraction process for a given request and parse the output."""
    result = extraction_chain.invoke({"request": request_text})
    # Assuming result.content contains the JSON string of goals and technologies
    try:
        # Convert JSON string to Python dictionary
        result_dict = json.loads(result.content)
        return result_dict
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response"}

# Example usage, if needed:
# request_text = "We have developed a chatbot that uses a transformer model for question answering."
# print(invoke_extraction(request_text))
