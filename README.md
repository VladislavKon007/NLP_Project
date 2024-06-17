

# Adapting BERT for Middle-Earth: Fine-Tuning Generalized NER Models with Domain-Specific Data from 'The Lord of the Rings'

## Abstract

This project explores enhancing BERT (Bidirectional Encoder Representations from Transformers) for Named Entity Recognition (NER) in specialized domains. We fine-tuned BERT using a dataset from J.R.R. Tolkien’s "The Lord of the Rings" (LotR) universe, scraped from the Eldamo website. Experiments included initial training on the English Web Treebank (EWT) dataset, fine-tuning on the LotR dataset, and combined training. Results show the LotR-trained model outperformed the EWT-trained model, achieving a peak F1 score of 0.8814 versus 0.5116. The combined approach demonstrated better generalization with an F1 score of 0.7369 on a unified test dataset. Our findings highlight the importance of domain-specific data and tokenization in improving NER performance for specialized contexts.

## Folder Structure

The project repository is organized as follows:

```
NLP_PROJECT
├── data
│   ├── .ipynb_checkpoints
│   ├── Plots
│   ├── Create_Plots.ipynb
│   ├── EWT_epochs_metrics.csv
│   ├── EWT_LOTR_metrics.csv
│   ├── EWT_LOTR_on_all_tags_metric.csv
│   ├── EWT_LOTR_on_all_tags_report.csv
│   ├── EWT_LOTR_reports.csv
│   ├── EWT_on_all_tags_metric.csv
│   ├── EWT_on_all_tags_report.csv
│   ├── EWT-training_metrics.csv
│   ├── EWT-training-report.csv
│   ├── final_metrics_per_label.csv
│   ├── LOTR_on_all_tags_metric.csv
│   ├── LOTR_on_all_tags_report.csv
│   ├── metrics.csv
│   ├── model-metrics-epoch.csv
│   ├── model-reports-epoch.csv
│
├── drafts
│   ├── bert_train_2-Copy1.ipynb
│   ├── bert_train_2.ipynb
│   ├── bert_train.py
│   ├── bert-try.py
│   ├── final_final.ipynb
│   ├── final_training-Copy1.ipynb
│   ├── final_training.ipynb
│   ├── final.py
│   ├── loading_model.ipynb
│   ├── loading_model_EWR_load.ipynb
│   ├── make_predictions.ipynb
│   ├── original.ipynb
│   ├── span_f1.py
│   ├── train_EWT_LOTR.ipynb
│   ├── train.ipynb
│   ├── try.py
│
├── README.md
├── tagged_sentences_test.iob2
├── tagged_sentences_train.iob2
├── train_EWT_LOTR_not_working.py
├── train_EWT_LOTR.ipynb
├── train_EWT.py
├── train_LOTR.py
```

## Project Description

The goal of this project is to adapt BERT for Named Entity Recognition (NER) tasks in a specialized domain using data from J.R.R. Tolkien's "The Lord of the Rings" universe. By fine-tuning BERT on this domain-specific dataset, we aim to enhance its ability to recognize and classify unique entities within this fantasy context.

### Data Collection and Preparation

- **Data Sources**: Eldamo website for Tolkien's invented languages and names, Kaggle for character names, and additional texts from "The Lord of the Rings."
- **Data Processing**: Scraped and categorized entities into characters, locations, and organizations. Converted data into IOB format for training.

### Experimental Setup

1. **Initial Training on EWT**:
   - Learning rate: 3e-5
   - Batch size: 64
   - Epochs: 7
   - Maximum sequence length: 174
   - Maximum gradient norm: 5

2. **Fine-Tuning on LOTR**: Same hyperparameters as initial training to ensure consistency.

3. **Combined Training**: Leveraged both general and domain-specific knowledge for a balanced model.

### Evaluation

- **Metrics**: F1 score, precision, and recall.
- **Datasets**: Evaluations conducted on LOTR Test Data and Combined EWT + LOTR Test Data.

## Training and Evaluation

To train and evaluate the model, run the respective Jupyter notebook files provided in the `drafts` directory. Here are the key notebooks to use:

- `bert_train_2.ipynb`
- `train_EWT_LOTR.ipynb`
- `final_training.ipynb`
- `make_predictions.ipynb`

## Results

- **LOTR-Trained Model**: Achieved an F1 score of 0.8814.
- **EWT + LOTR-Trained Model**: Demonstrated better generalization with an F1 score of 0.7369 on a unified test dataset.

## Conclusion

Our experiments highlight the importance of domain-specific data and tokenization in improving the performance of NER models in specialized contexts. Fine-tuning BERT on a dataset derived from "The Lord of the Rings" universe significantly enhanced its ability to recognize and classify unique entities.

## Authors

- **Costel Gutu**
- **Oleksandr Adamov**
- **Vladislav Konjusenko**

For more details, please refer to the full project report.

---

Feel free to add or modify any details as necessary.