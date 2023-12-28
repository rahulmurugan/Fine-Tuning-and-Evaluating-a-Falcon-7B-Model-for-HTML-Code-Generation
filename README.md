


# Fine-Tuning and Evaluating a Falcon 7B for HTML Code Generation - 
### OBJECTIVE
Fine-tune the above model or any other  open-source language model of your choice to generate HTML code from given natural language prompts. This assignment focuses on demonstrating your skills in model fine-tuning, handling potential challenges, and basic API development.


## Installation

### Required Packages






### Model

ybelkada/falcon-7b-sharded-bf16 (Imported from huggingface)

#### Features:
The model "ybelkada/falcon-7b-sharded-bf16" refers to a pre-trained language model known as Falcon 7B, which utilizes sharding and 16-bit (BF16) precision during training. Here are some key features and aspects commonly associated with this model:

1.Falcon Model: The Falcon series of models are designed to be highly efficient language models, often optimized for speed, memory, and performance on various natural language processing (NLP) tasks.

2.Sharded Architecture: Sharding refers to splitting a large model across multiple processing units or devices. Falcon 7B uses sharding, dividing the model's components across different computational units to enable efficient parallel processing.

3.16-bit Precision (BF16): BF16, or bfloat16, represents a 16-bit floating-point number format that aims to strike a balance between speed and precision. It provides faster computation compared to 32-bit floating-point representation while maintaining sufficient accuracy for many NLP tasks.

4.Efficiency: Falcon models like Falcon 7B are often optimized for low-latency inference, making them suitable for deployment in real-time applications or scenarios requiring rapid responses.

5.Fine-Tuning: The model can be fine-tuned on domain-specific datasets to adapt its knowledge to particular tasks, such as text generation, summarization, or language understanding.

6.Language Understanding and Generation: Falcon 7B can be used for a wide range of NLP tasks, including text generation, summarization, question answering, and language translation, among others.

7.Large-Scale Capability: It's built on a large-scale architecture, allowing it to handle a diverse range of language-related tasks and understand complex linguistic patterns.


    
## Data Set
Used the ttbui/html_alpaca from Hugging Face Datasets which contains pairs of natural language prompts and corresponding HTML code.
## 

![App Screenshot](https://snipboard.io/rbLofI.jpg)



## Why Quantization of Falcon 7b ?

### Quantization Configuration Explanation

#### BitsAndBytes Quantization
Quantization is a technique used to optimize the model's computational efficiency and reduce memory footprint without compromising much on performance. In this code snippet, the model undergoes BitsAndBytes (BNB) quantization for better efficiency.

- **Load in 4-bit**: Enabling this parameter initiates the usage of 4-bit precision instead of conventional 32-bit or 16-bit precision. It's a technique to reduce the number of bits used to represent each weight or activation.
  
- **4-bit Quantization Type (NF4)**: The chosen 4-bit quantization type here is "NF4," which stands for Normal Float 4-bit. NF4 introduces a new data type, 4-bit normal float, aiming to strike a balance between 4-bit integers and 4-bit floating-point representations. This choice of quantization type often yields improved results compared to other 4-bit formats.

- **4-bit Compute Data Type (torch.float16)**: This parameter specifies the data type used for computation during 4-bit quantization. Here, the choice is torch.float16, which is a 16-bit floating-point representation used for computations involving quantized values.

#### Purpose of Quantization
Quantization, especially using 4-bit precision and innovative data types like NF4, helps to significantly reduce model size and memory usage. This reduction in precision allows for more efficient model deployment on hardware with lower computational capabilities, such as mobile devices or edge devices, without sacrificing performance drastically.

#### Loading the Falcon Model with Quantization Configuration
The Falcon model ("ybelkada/falcon-7b-sharded-bf16") is loaded with the BitsAndBytes quantization configuration. This means the model will be fine-tuned or used for inference with the specified quantization settings enabled, enhancing its efficiency for various natural language processing tasks.

## Techniques for Quantization

## 

![App Screenshot](https://snipboard.io/ck29NJ.jpg)

## Preparing the Data Set
### Loading the Dataset
The code snippet initiates by loading the dataset titled "ttbui/html_alpaca" specifically for the "train" split using the Hugging Face Datasets library.

### Column Removal
After loading the dataset, specific columns ('input' and 'response') are removed from the dataset. This step could be attributed to several reasons, such as these columns containing irrelevant or sensitive information or the exclusion of certain features for the intended task.

### Tokenization with AutoTokenizer
Utilizing the AutoTokenizer from the Hugging Face Transformers library, the code initializes a tokenizer using the provided 'model_name' for further tokenization processes. Tokenizers convert raw text data into tokenized sequences suitable for model input.

### Mapping and Tokenization
The loaded dataset is mapped to apply a custom tokenize_function to each example within the dataset. This function concatenates 'instruction' and 'output' data and tokenizes it using the initialized tokenizer, ultimately transforming the text into tokenized sequences.

### Final Dataset Preparation
After tokenization, columns 'input' and 'response' are removed from the tokenized dataset, indicating that these columns are no longer necessary for subsequent steps. The resulting dataset is ready for further preprocessing or direct utilization in training or evaluation phases.

### Splitting into Training and Testing Sets
The final tokenized dataset is split into training and testing subsets, with an 80-20 ratio, using the train_test_split function. The 'test_size=0.2' parameter determines an 80% training and 20% testing split, aiding in model training and evaluation.

Each step in this data preparation process contributes to transforming raw textual information into a structured format suitable for subsequent machine learning tasks such as fine-tuning models or training.
## Fine Tuning LLMs

Fine-tuning assists in maximizing the potential of pre-trained large language models (LLMs) by adjusting their weights to align more closely with specific tasks or domains. This approach yields higher-quality results compared to conventional prompt engineering, and it significantly reduces costs and latency. In this guide, we'll delve into the fundamentals of LLM fine-tuning and explore the initiation process using Modal's cutting-edge techniques.            
Cost Benefits
Fine-tuning a Language Model (LLM) offers several advantages in terms of cost efficiency:

Effectiveness over Prompting: Compared to using prompts, fine-tuning is generally more effective and efficient in directing an LLM's behavior.

Optimization of Inputs: Training the model on a specific dataset enables the reduction of input tokens needed for effective performance without compromising quality.

Usage of Smaller Models: Fine-tuning often allows the utilization of smaller models, leading to decreased latency and inference costs while maintaining comparable performance.

Reduces GPU Dependency 

### Steps for Fine-Tuning
- Choose a Base Model: Select the base language model that best fits your task requirements.

- Prepare the Dataset:

   Create the Prompt: Develop a well-crafted set of examples or prompts that represent the desired task.
  
  Tokenize the Prompt: Convert the prompt into tokens to   prepare it for model input.
- Training: Fine-tune the selected base model on the prepared dataset to adapt its behavior to the specific task or domain.

- Using Advanced Fine-Tuning Strategies:

  Parameter-Efficient Fine-Tuning (PEFT): Employ advanced fine-tuning techniques like PEFT to optimize model parameters efficiently. PEFT implements a number of techniques that help aims to reduce the memory requirements while speeding up fine-tuning by freezing most of the parameters and only training a subset of the parameters. The most popular PEFT technique is Low-Rank Adaption (LoRA). Instead of tweaking the original weight matrix directly, LoRA simply updates a smaller matrix on top, the “low-rank” adapter. This small adapter captures the essential changes needed for the new task, while keeping the original matrix frozen. To produce the final results, you combine the original and trained adapter weights.

## LoRA

The beauty of LoRA, or Low-Rank Adaptation, lies in its elegant simplicity and its profound impact on the training of Large Language Models (LLMs). The core concept of LoRA revolves around an equation that succinctly summarizes how a neural network adapts to specific tasks during training. Let’s break down this equation:

Ouput(h)= Input(x)Pre-trained weight(W) + Input(x)LowRankUpdate(BA)
#
In this equation, the terms represent:
Output (h): This is the result of our operation. It represents the processed data after it has been run through a layer of the neural network.
Input (x): This represents the data being input into the layer of the neural network for processing.
PretrainedWeight (W₀): This represents the pre-existing weights of the neural network. These weights have been learned over time through extensive training and are kept static during the adaptation process, meaning they do not receive gradient updates.
LowRankUpdate(BA): This is where the true magic of LoRA comes in. LoRA assumes that the update to the weights during task-specific adaptation can be represented as a low-rank decomposition, where matrices B and A encapsulate these updates in a more compact, memory-efficient manner.

![App Screenshot](https://snipboard.io/yVlQO2.jpg)
## Training and Fine Tuning
### Enabling Gradient Checkpointing and Model Preparation


Enabling Gradient Checkpointing and LORA Preparation
Before training the model, gradient checkpointing is enabled to optimize memory usage and prepare the model for Low-Rank Adaptation (LORA) fine-tuning

### Configuration of LORA Hyperparameters

LORA (Low-Rank Adaptation) settings are established to fine-tune the model. These settings include:

LORA Alpha: Controls the scaling factor for low-rank adaptation, influencing the impact of the low-rank matrix on model weights.

LORA Dropout: Regulates the dropout rate during LORA to prevent overfitting and enhance model generalization.

LORA Rank: Specifies the dimensionality of the low-rank matrix, affecting memory and computational requirements.

LORA Bias: Determines the inclusion or exclusion of bias terms in the adaptation.

Task Type: Defines the specific task type the model is being adapted for.

Target Modules: Identifies neural network components for targeted adaptation.

### Training Configuration

The model training settings, including hyperparameters and configurations, are defined as follows:

Gradient Accumulation Steps: Aggregates gradients across multiple steps before updating model parameters, beneficial for large batch sizes.

Per Device Train Batch Size: Determines the number of samples processed per device in each training step.

Learning Rate: Controls the step size during optimization, influencing the model's parameter updates.

FP16: Enables mixed-precision training by utilizing 16-bit floating-point format, reducing memory usage.

Save Total Limit: Specifies the maximum number of checkpoints to save during training.

Logging Steps: Sets the frequency of logging training metrics.
Output Directory: Specifies the location to save training checkpoints.

Save Strategy: Determines when to save checkpoints during training.

Optimizer: Chooses the optimization algorithm used during training (e.g., "paged_adamw_8bit").

LR Scheduler Type: Specifies the learning rate scheduling strategy.

Warmup Ratio: Controls the ratio of warmup steps in the learning rate schedule.

### Initialization and Training

The Trainer is initialized with the configured model, dataset, and training arguments, commencing the training process.

This setup allows for customized training with specific LORA adaptations and tailored training configurations, impacting the model's efficiency and performance for the target task or dataset. Adjusting these parameters optimizes training dynamics and the resulting model's effectiveness.
## Training Result

![App Screenshot](https://snipboard.io/Vd1AY7.jpg)
## Model Inference and Evaluation

This section encompasses the evaluation and inference process of the trained model against a test dataset. It involves the following steps:

-  Saving the Trained Model
The trained model is saved using the trainer.save_model() function, preserving the model's state for future use.

-  Evaluation and Prediction Loop
Function Definition: Define an inference function (inference) responsible for generating text based on a given prompt using the model and tokenizer. Ensure it tokenizes the input prompt and generates the output text.
Function to Check Exact Match: Defines a function, is_exact_match(a, b), to compare predicted and target answers to determine if they match perfectly.

Inference Function: Establishes a function, inference(text, model, tokenizer, max_input_tokens, max_output_tokens), to generate predictions based on input text using the trained model and tokenizer.
- Evaluation Process
Loading the Evaluation Dataset: Loads the evaluation dataset (replace split_dataset["test"] with the path to the test dataset).
Iterating Over the Test Dataset: Iterates through the test dataset to perform inference using the trained model.

Calculating Exact Matches: Checks for exact matches between predicted and target answers and maintains a count of these matches to assess model accuracy.
Creating a DataFrame of Predictions: Creates a DataFrame df containing predicted and target outputs for analysis and further inspection.
- Loss Function
Loss Calculation Function: Create a function (calculate_loss) to compute the loss between the predicted and target outputs. This function should take in tokenized predicted and target outputs, calculate the loss (e.g., using torch.nn.CrossEntropyLoss), and return the loss value.
- Result Display
Displaying Evaluation Metrics: Prints the total count of exact matches obtained during evaluation.
Printing DataFrame of Predictions: Outputs a DataFrame df showing the predicted and target outputs for each example in the evaluation dataset.

![App Screenshot](https://snipboard.io/DnGX85.jpg)
## API Development
## Building a Local API using Flask for Model Inference

### Description

This section outlines the steps to set up a local API using Flask to perform model inference. We'll assume you have a trained model (`final_model`) and you want to use it to generate HTML based on provided prompts.

### Prerequisites


- Required Python packages (`flask`, `requests`, etc.) installed

### Procedure

1. **Setup and Configuration:**
   - Define the required directories (`output_dir`, `offload_folder`) and the trained model (`final_model`).

2. **Flask Application:**
   - Create a Flask application to handle incoming requests and perform model inference based on the provided prompts.
   - Start the Flask app on a specified port (e.g., `5001`) in the background using a separate thread.

3. **Send Requests to Flask API:**
   - Define the prompt for which you want to generate HTML.
   - Send a POST request to the Flask API (`http://127.0.0.1:5001/generate_html`) with the prompt using the `requests` library.
   
4. **Receive and Handle Responses:**
   - Capture the response from the Flask API and print the generated HTML or any errors encountered during the inference process.

### Usage

1. **Starting the Flask Application:**
   - Run the Python script that sets up the Flask API.
   - Verify that the API is running on `http://127.0.0.1:5001`.

2. **Sending Requests to the API:**
   - Use `requests.post` to send prompts to the Flask API and retrieve the generated HTML.
