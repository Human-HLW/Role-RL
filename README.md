# Role-Based Reinforcement Learning for LLM Selection

This project implements a Role Reinforcement Learning (RL) framework to dynamically select the most suitable Large Language Models (LLMs) for different roles in a text processing pipeline. The goal is to optimize the performance (reward) and cost of using LLMs for Online Long-Transcript Processing (OLP) including topic finding, topic locating, relationship checking, and content organization.

## Overview

The program uses a Q-learning algorithm to learn the best LLM for each role based on the rewards and costs associated with their performance. The framework is designed to handle multiple tasks and adapt to changes in task difficulty over time.

## Key Components

### LLM Selection:
The program selects from a predefined list of LLMs, each with associated input and output costs.

### Roles:

- **Topic Finder**: Identifies items being sold in a given text.
- **Topic Locator**: Locates the sentences related to each identified item.
- **Relationship Checker**: Checks if adjacent items are the same and merges them if necessary.
- **Content Organizer**: Organizes the content related to each item into predefined categories.

### Reinforcement Learning:
The program uses Q-learning to update the selection strategy for each role based on the rewards and costs of previous selections.

### Board Member Election:
A Markov chain-based election process is used to update the weights of the LLMs acting as board members for judging the correctness of responses.

## Requirements

- Python 3.x
- Libraries: `requests`, `numpy`, `pandas`, `matplotlib`, `Levenshtein`, `json`, `ast`, `datetime`, `random`, `re`, `time`

## Installation
### Clone the repository:

```bash 
git clone https://github.com/XXX/Role-RL.git
cd Role-RL
```

### Install the required libraries:

```bash 
pip install -r requirements.txt
```

## Usage
### Configuration:

Modify the LLMs list in the script to include the LLMs you want to use, along with their input and output costs.

Set the url and headers for the API endpoint you are using to interact with the LLMs.

### Running the Program:

```bash
python main_Role_RL.py
```

### Output:

The program will output the history of LLM selections, rewards, and costs for each role.

It will also generate an Excel file (History_YYYYMMDD_HHMMSS.xlsx) containing detailed logs and Q-tables for each role.

Plots showing the LLM selection, rewards, and costs over time will be displayed.

## Parameters

- `iters`: Total number of iterations (default: 50000).
- `epsilon`: Exploration rate for the RL algorithm (default: 0.03).
- `alpha`: Learning rate for the RL algorithm (default: 0.1).
- `gamma`: Discount factor for the RL algorithm (default: 0).
- `change_hardness_last`: Number of iterations for each task difficulty level (default: 500).
- `prob_change_hardness`: Probability of changing the task difficulty (default: 0.03).

## Files

- `main_Role_RL_comment.py`: Main script implementing the RL framework.
- `Task1.txt`, `Task2.txt`, `Task3.txt`: Example task files containing text data for processing.
- `output_dict_judged_by_Gem.txt`, `output_dict_judged_by_Cla.txt`: Output of the example tasks, evaluated by two board members.
- `parameters.txt`: File to store the print limit parameter.

## Customization

- **Task Files**: You can add or modify the task files (`Task1.txt`, `Task2.txt`, `Task3.txt`) to include different text data for processing.
- **LLM List**: Modify the LLMs list to include different LLMs or update their costs.
- **Roles**: You can add or modify the roles and their corresponding functions in the script.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The project uses the Levenshtein library for string similarity calculations.
- The reinforcement learning framework is inspired by traditional Q-learning algorithms.
