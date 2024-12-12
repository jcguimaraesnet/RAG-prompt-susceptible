# RAG-prompt-susceptible - How susceptible are LLMs to syntactic variations of the same prompt on RAG systems?

This repository contains the code, data, and analysis for our study [link later] on advanced recovery augmented generation (RAG) techniques. It is part of our scientific article that investigates how susceptible LLMs are to syntactic variations of the same prompt in RAG systems.

## Repository structure

- `eval_questions/`: Contains a JSON file with 107 QA pairs used in the assessment.
- `resources/`: Includes essential resources such as prompt template and configuration files. Note: The actual configuration files need API keys and other settings to be populated.
- `main.py`: The main script where experiments are defined and run.
- `utils.py`: Helper functions that support various operations within the repository.
- `indexing_final`: Scripts to configure vector databases, phrase window and document summary.

## Getting started

To replicate our experiments or analyze our results, be sure to fill in the required API keys and other settings by creating a `. env` (see `. sample. env`) - the `. env` is ignored in . gitignore for security.

Configure the python environment using `venv` or `pyenv` or your favorite python environment manager. Call the environment `RAG-prompt-susceptible` or whatever you want.
- `python3 -m venv RAG-prompt-susceptible` and enable it using `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows).
- OR `pyenv` with `pyenv virtualenv 3.12 RAG-prompt-susceptible` and enable with `pyenv local RAG-prompt-susceptible`.

Then run `pip install -r requirements.txt` to install all required dependencies.

## Full replication

To configure vector databases for experiments, run the `indexing_final.py` script. Afterwards, run `upload-benchmark-dev.py` to upload the questions to the tonic validation platform. Then, run `main.py` to perform the experiments. Helper functions in `utils.py` and `custom_posrewriting_query_engine.py` are employed in scripts to speed up processes.

## Contribution

Contributions are welcome. For any changes or improvements, please first open an issue to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).