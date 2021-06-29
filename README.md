# Usage

## Setup
1. Create a new environment:

```bash
conda create --name disrpt python=3.8
conda activate disrpt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download all fastText embeddings 
([link](https://drive.google.com/file/d/1HUiwJheEn4QfYeLrcazm-HDdNDT86gmK/view?usp=sharing))
and put them in the `embeddings/` folder.

4. Download full data, with underscores restored, and place it in `data/2021` for the 2021 shared task or 
`data/2019` for the previous shared task. This should result in paths like 
`data/2021/deu.rst.pcc/deu.rst.pcc_train.rels`.

Note: if you're a member of GU Corpling, you can download zips for the 
[2019](https://drive.google.com/file/d/1fkGTBJT7C--vfINi-iEY-6RJQ9HsRXoX/view?usp=sharing) 
and [2021](https://drive.google.com/file/d/1cefWFFaO9Hb4yuONSdbelVVDausotKLD/view?usp=sharing) data.

## Experiments

See `seg_scripts` and `rel_scripts`.
