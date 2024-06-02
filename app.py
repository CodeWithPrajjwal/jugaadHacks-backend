from julep import Client
from flask import Flask, request, jsonify
from julep import Client
from flask_cors import CORS
import pickle
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate
from tensorflow.keras.models import Model
import re
import os

app = Flask(__name__)
CORS(app)
load_dotenv(".env")

api_key = os.environ.get("JULEP_API_KEY")
base_url = os.environ.get("JULEP_API_URL")


class DrugInteractionModel:
    def __init__(self, data_path, embedding_dim=100, lstm_units=64, max_vocab_size=32000):
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_vocab_size = max_vocab_size
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size)
        self.model = None
        self.max_sequence_length = 0

    def load_data(self):
        with open(self.data_path, "rb") as f:
            self.LabelledData = pickle.load(f)

    def preprocess_data(self):
        def extract_drug_names(data, index):
            return [key[index] for key in data.keys()]

        def extract_interaction_labels(data):
            return [value for value in data.values()]

        # Extract data
        self.drug1_names = extract_drug_names(self.LabelledData, 0)
        self.drug2_names = extract_drug_names(self.LabelledData, 1)
        self.interaction_labels = extract_interaction_labels(self.LabelledData)

        # Convert interaction labels to numeric
        interaction_label_mapping = {"unsafe": 1, "safe": 0}
        self.interaction_labels = [interaction_label_mapping[label] for label in self.interaction_labels]

        # Tokenize the drug names
        self.tokenizer.fit_on_texts(self.drug1_names + self.drug2_names)

        # Convert drug names to integer sequences
        drug1_sequences = self.tokenizer.texts_to_sequences(self.drug1_names)
        drug2_sequences = self.tokenizer.texts_to_sequences(self.drug2_names)

        # Determine the maximum sequence length for padding
        self.max_sequence_length = max(len(seq) for seq in drug1_sequences + drug2_sequences)

        # Pad the sequences
        padding_token = 0
        self.drug1_padded = pad_sequences(drug1_sequences, maxlen=self.max_sequence_length, padding="post", value=padding_token)
        self.drug2_padded = pad_sequences(drug2_sequences, maxlen=self.max_sequence_length, padding="post", value=padding_token)

        # Convert padded sequences and interaction labels to numpy arrays
        self.drug1_padded_np = np.array(self.drug1_padded)
        self.drug2_padded_np = np.array(self.drug2_padded)
        self.interaction_labels_np = np.array(self.interaction_labels)

    def build_model(self):
        # Define the model using functional API
        input1 = Input(shape=(self.max_sequence_length,))
        input2 = Input(shape=(self.max_sequence_length,))

        embedding_layer = Embedding(len(self.tokenizer.word_index) + 1, self.embedding_dim)

        embedded1 = embedding_layer(input1)
        embedded2 = embedding_layer(input2)

        # Use LSTM layer on both embedded inputs
        lstm_out1 = LSTM(self.lstm_units)(embedded1)
        lstm_out2 = LSTM(self.lstm_units)(embedded2)

        # Concatenate the outputs of both LSTM layers
        concatenated = Concatenate()([lstm_out1, lstm_out2])

        output = Dense(1, activation="sigmoid")(concatenated)

        # Create the model
        self.model = Model(inputs=[input1, input2], outputs=output)

        # Compile the model
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def train_model(self, epochs=10):
        # Fit the model
        self.model.fit(x=[self.drug1_padded_np, self.drug2_padded_np], y=self.interaction_labels_np, epochs=epochs)

    def predict_interaction(self, drug1, drug2):
        # Tokenize and pad the input drug names
        drug1_seq = self.tokenizer.texts_to_sequences([drug1])
        drug2_seq = self.tokenizer.texts_to_sequences([drug2])

        drug1_padded = pad_sequences(drug1_seq, maxlen=self.max_sequence_length, padding="post", value=0)
        drug2_padded = pad_sequences(drug2_seq, maxlen=self.max_sequence_length, padding="post", value=0)

        # Predict the interaction
        prediction = self.model.predict([drug1_padded, drug2_padded])
        return "unsafe" if prediction[0][0] > 0.5 else "safe"



model = DrugInteractionModel(data_path="LabelledData.pkl")
model.load_data()
model.preprocess_data()
model.build_model()
model.train_model(epochs=10)

client = Client(api_key=api_key, base_url=base_url)

def retrieve_createTitleSummaryAgent():
    agents = client.agents.list(metadata_filter={"role": "summarization"})
    if not agents:
        print("[!] Creating a new text summarization agent")
        agent = createTitleSummaryAgent()
        return agent
    else:
        return agents[0]

def retrieve_createQueryHandlingAgent(instruction : list):
    agents =   client.agents.list(metadata_filter={"role": "query_handling"})
    if not agents:
        print("[!] Creating a new query handling agent")
        agent =   createQueryHandlingAgent(instruction)
        return agent
    else:
        return agents[0]
    
def retrieve_createDrugExtractionAgent():
    agents =   client.agents.list(metadata_filter={"role": "drug_extraction"})
    if not agents:
        print("[!] Creating a new drug extraction agent")
        agent =   createDrugExtractionAgent()
        return agent
    else:
        return agents[0]

def createTitleSummaryAgent():
    agent =   client.agents.create(
        name="TitleSummary",
        about="This agent generates a summary of the given text.",
        default_settings={
            "temperature": 0.7,
            "top_p": 1,
            "min_p": 0.01,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "length_penalty": 1.0,
        },
        instructions=["You have to generate a summary in about 3-4 words of the medical query prompted by the user."],
        model="gpt-4o",
        metadata={"role" : "summarization"}
    )
    return agent

def createQueryHandlingAgent(instruction : list):
    agent =   client.agents.create(
        name="QueryHandling",
        about="This agent helps in handling the medical queries of the user, about the symptoms and possible drug interactions.",
        default_settings={
            "temperature": 0.7,
            "top_p": 1,
            "min_p": 0.01,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "length_penalty": 1.0,
        },
        instructions=instruction,
        model="gpt-4o",
        metadata={"role" : "query_handling"}
    )
    return agent

def createDrugExtractionAgent():
    agent =   client.agents.create(
        name="DrugExtraction",
        about="This agent helps in extracting the drug names from the given user prompt.",
        default_settings={
            "temperature": 0.7,
            "top_p": 1,
            "min_p": 0.01,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "length_penalty": 1.0,
        },
        instructions=["You have to only extract the names of the drugs from the given user prompt if exists.",
        "choose two drugs that can be unsafe if they be used together out of the user prompts",
        "only output a python list containing two drugs if not return an empty list"],
        model="gpt-4o",
        metadata={"role" : "drug_extraction"}
    )
    return agent

def retrieve_TitleSummarySession(agent_id: str, user_id: str):
    sessions =   client.sessions.list(
        metadata_filter={"db_uuid": "1234"}
    )
    if not sessions:
        print("[!] Creating a new title summary session")
        session =   create_TitleSummarySession(agent_id, user_id)
        return session
    else:
        return sessions[0]

def retrieve_QueryHandlingSession(agent_id: str, user_id: str):
    sessions =   client.sessions.list(
        metadata_filter={"db_uuid": "4567"}
    )
    if not sessions:
        print("[!] Creating a new query handling session")
        session =   create_QueryHandlingSession(agent_id, user_id)
        return session
    else:
        return sessions[0]
    
def retrieve_DrugExtractionSession(agent_id: str, user_id: str):
    sessions =   client.sessions.list(
        metadata_filter={"db_uuid": "8910"}
    )
    if not sessions:
        print("[!] Creating a new drug extraction session")
        session =   create_DrugExtractionSession(agent_id, user_id)
        return session
    else:
        return sessions[0]


def create_TitleSummarySession(agent_id: str, user_id: str):
    session =   client.sessions.create(
        user_id=user_id,
        agent_id=agent_id,
        situation="User has created a new chat session for his concerns about the symptoms he is facing. You have to generate a summary of the medical query prompted by the user.",
        metadata={"db_uuid": "1234"}
    )
    return session

def create_QueryHandlingSession(agent_id: str, user_id: str):
    session =   client.sessions.create(
        user_id=user_id,
        agent_id=agent_id,
        situation="You are a medical professional who is handling the medical queries of the user, about the symptoms and possible drug interactions.",
        metadata={"db_uuid": "4567"}
    )
    return session

def create_DrugExtractionSession(agent_id: str, user_id: str):
    session =   client.sessions.create(
        user_id=user_id,
        agent_id=agent_id,
        situation="You are a medical professional who is handling the medical queries of the user, about the symptoms and possible drug interactions.",
        metadata={"db_uuid": "8910"}
    )
    return session

def retrieve_user():
    users =   client.users.list(metadata_filter={"name": "John"})
    if not users:
        print("[!] Creating a new user")
        user =   create_user()
        return user
    else:
        return users[0]


def create_user():
    user =   client.users.create(
        name="John",
        about="User is seeking medical advice for their symptoms. Possible drug interactions are to be provided. Cheap and effective solutions are to be provided.",
        metadata={"name": "John"}
    )
    return user

def get_drug_interaction_prediction(model,drug1 : str, drug2 : str):
    # return "The interaction between {} and {} is: {}".format(drug1, drug2, "No interactions found")
    result = model.predict_interaction(drug1, drug2)
    return result

@app.route('/chat', methods=['POST'])
def chat():
    prompt = request.json['prompt']
    new_chat = request.json['new_chat']

    response = {"summary": "", "drug_extraction": "", "result": ""}

    safe_interaction_instruction = ["You necessarily have to provide the user with a message that the drug interaction between the drugs is safe.",
                                    "Provide the user with a list of possible reasons for the symptoms they are facing.",
                                    "Provide the user with a list of possible solutions for the symptoms they are facing."]
    unsafe_interaction_instruction = ["You necessarily have to provide the user with a message that the drug interaction between the drugs is unsafe.",
                                    "Provide the user with a list of possible reasons for the symptoms they are facing.",
                                    "Provide the user with a list of possible solutions for the symptoms they are facing."]
    unknown_interaction_instruction = ["Start a conversation with user and ask for their symptoms and the drugs they are taking if not provided already",
            "Study the user's input & check the salt composition of the drugs they are taking",
            "If no drug interactions are found, provide the user with a list of cheap and effective solutions for the symptoms they are facing",
            "Provide the user with a list of possible reasons for the symptoms they are facing if no drug interactions are found",
            "Provide the user with a list of possible solutions for the symptoms they are facing if no drug interactions are found",
            "End the conversation by asking the user if they need any more help"]

    user =   retrieve_user()

    if new_chat:
        summary_agent =   retrieve_createTitleSummaryAgent()
        summary_session =   retrieve_TitleSummarySession(summary_agent.id, user.id)

        text_summary =   client.sessions.chat(
            session_id=summary_session.id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "name": "John",
                }
            ],
            recall=True,
            remember=True,
        )
        response['summary'] = text_summary.response[0][0].content
    
    drug_extraction_agent =   retrieve_createDrugExtractionAgent()
    drug_extraction_session =   retrieve_DrugExtractionSession(drug_extraction_agent.id, user.id)

    drug_extraction =   client.sessions.chat(
        session_id=drug_extraction_session.id,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "name": "John",
            }
        ],
        recall=True,
        remember=True,
    )
    response['drug_extraction'] = drug_extraction.response[0][0].content
    print(response['drug_extraction'])

    pattern = r"\[([^\]]*)\]"  # Matches characters except ']' inside brackets

# Use re.findall to extract all matches (in case of nested lists)
    drug_list_matches = re.findall(pattern, response['drug_extraction'])

# If there's only one match (assuming a single list), use the first element
    if len(drug_list_matches) == 1:
      drug_list = eval(drug_list_matches[0])  # Use eval cautiously   

      if get_drug_interaction_prediction(model,drug_list[0], drug_list[1]) == "safe":
        instruction = safe_interaction_instruction
        queryhandling_agent =   retrieve_createQueryHandlingAgent(instruction)
        queryhandling_session =   retrieve_QueryHandlingSession(queryhandling_agent.id, user.id)

        safe_interaction_result =   client.sessions.chat(
            session_id=queryhandling_session.id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "name": "John",
                }
            ],
            recall=True,
            remember=True,
        )

        response['result'] = safe_interaction_result.response[0][0].content

      elif get_drug_interaction_prediction(model,drug_list[0], drug_list[1]) == "unsafe":
        instruction = unsafe_interaction_instruction
        queryhandling_agent =   retrieve_createQueryHandlingAgent(instruction)
        queryhandling_session =   retrieve_QueryHandlingSession(queryhandling_agent.id, user.id)

        unsafe_interaction_result =   client.sessions.chat(
            session_id=queryhandling_session.id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "name": "John",
                }
            ],
            recall=True,
            remember=True,
        )
        response['result'] = unsafe_interaction_result.response[0][0].content
    else: 
        instruction = unknown_interaction_instruction
        queryhandling_agent =   retrieve_createQueryHandlingAgent(instruction)
        queryhandling_session =   retrieve_QueryHandlingSession(queryhandling_agent.id, user.id)

        unsafe_interaction_result =   client.sessions.chat(
            session_id=queryhandling_session.id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "name": "John",
                }
            ],
            recall=True,
            remember=True,
        )
        response['result'] = unsafe_interaction_result.response[0][0].content
    
    response.headers['Access-Control-Allow-Origin'] = 'https://frontend-jugaadhacks.onrender.com'  # Replace with your actual frontend URL
    response.headers['Access-Control-Allow-Credentials'] = 'true'  # Allow cookies if needed  
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)
