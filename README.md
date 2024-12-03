# LoRT: Logical Reasoning Evaluation Suite for Transformers

This repository contains the LoRT dataset that was introduced at the CMSC 828J course offered at University of Maryland, Colege Park.

Full details about the paper can be found in our Findings paper: [LoRT: Logical Reasoning Evaluation Suite for Transformers]()

## Introduction
**LoRT** is a large, manually-curated set of CommonsenseQA-style questions for probing logical reasoning capabilities in Language models especially the transformer based models. 

All questions were written in **English**.

## Dataset
The dataset is released in both `JSON` and `CSV` formats. FORK contains `110K` CommmonsenseQA-style questions. 

The following is an example entry from FORK:
```
{
  "Given {A} contradicts {B}, does {A} forward entail {B}?",
    "No"
}
```
A description of the fields is given below:
* `A`: the first word. It can be real or synthetic.
* `B`: the second word. It can be real or synthetic.
* `No`: the answer to the entailment question asked.
## License
This project is licensed under the CC-BY-4.0 License.
## Citation
If you use this dataset, please cite the following paper:
```
@inproceedings{wu-sarkar-yarmovsky-2024-lort,
    title = "{LoRT}: Logical Reasoning Evaluation Suite for Transformers",
    author = "Wu, Wenshan, Sarkar, Aditya  and Yarmovsky, Jenny",
    booktitle = "Course Project for CMSC 828J",
    month = dec,
    year = "2024",
    address = "College Park, US"
}
```
This repo will be legend - wait for it - ary.. legendary!
