# Automatic Term Extraction API

## 1. Description
In this repo, we wrote an API to inference SloBERTa term extractor, whose model has been trained with [RSDO5 corpus](https://www.clarin.si/repository/xmlui/handle/11356/1400). Feel free to check out this [repo](https://github.com/honghanhh/ate-2022) for better understanding about the methodology.

---

## 2. Requirements

Please install all the necessary libraries noted in [requirements.txt](./requirements.txt) using this command:

```
pip install -r requirements.txt
```

## 3. Implementation

Run the following command on the terminal:

```python
main.py
```

A link will be show on the terminal so that you can access to the API and test requests.
```python
 * Serving Flask app 'main'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 185-520-689
```

We suggest you use POSTMAN or Hoppscotch to test the API with an uploaded file as the output whose format is CONLL-like (plese check out [temp_1.conllu](temp_1.conllu)). See the demonstration as the image below.

![demo](./imgs/demo.png)


## 4. Docker version
Run the following command:
```python
docker build -t ate .  
docker run -d -p 5000:5000 ate
```

## Contributors:
- 🐮 [TRAN Thi Hong Hanh](https://github.com/honghanhh) 🐮
