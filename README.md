# Identification of writers by handwriting.
### Authors: *Samuel Kiszka, David Klajbl, Matúš Pestun*
### Supervisor: *Ing. Jan Kohút*

### Project file structure:
```text
knn-writer-identification/
│
├── doc/                    # Documentation
|
├── singularity/            # Singularity environment setup utilities
│
├── src/                    # Implementation
│   │
│   ├── model.py                # Proposed model
│   ├── model_baseline.py       # Baseline model
│   ├── train_id_embedding.py   # Model training script
│   ├── test_model.py           # Model testing script
│   ├── id_dataset.py           # Pytorch dataset class
│   ├── eval/                   # Model evaluation utilities
│   ├── patchers/               # Image patching utilities
│   └── ...                     # Other helper utilities
│
└── requirements_*.txt      # Python dependencies
```
### Acknowledgements
Some parts of this project are based on code provided by the supervisor Ing. Jan Kohút and were modified for use in this project.

### AI Assistance Disclaimer
Some parts of this project’s source code were developed with assistance from AI tools.
