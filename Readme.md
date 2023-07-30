Some parts of the code, came from the author's original [repository](https://github.com/lingochamp/Multi-Scale-BERT-AES#readme).

The original code only provides the function of predicting according to the pre-trained weights provided by the author.

I fixed the unnecessary parts of the original code and added the traning part following the procedure in the paper.

Experimental results from the newly trained model will be added soon.

---

This is a demo for the paper(NAACL 2022 long paper accepted): "On the Use of BERT for Automated Essay Scoring: Joint Learning of Multi-Scale Essay Representation"

Steps to run the decoding process:
step 1: Download the project to the computer.
step 2: Download the model(for ASAP's prompt 8) to the computer with the following link:
Link: https://pan.baidu.com/s/1_m_-DQlX-dLh1XdhOMzj1A?pwd=tmmb
step 3: Update the config file "asap.ini" and set the parameters as following:
data_dir: your local directory to store the essays of prompt 8 (file format: id \t text \t score);
model_directory: your local directory to store the model which is obtained in step 2;
result_file: the path of file to store prediction result.
step 4: Run the following script to get scoring result:
sh decode.sh
