# SpaCy Model Training & Upgrading Guide 

* Create **base_config.cfg** file on spaCy's widget. (https://spacy.io/usage/training)
* Initiate remaining config by following command:

```shell
python -m spacy init fill-config base_config.cfg config.cfg
```

* Prepare spaCy 2.0 training data format, the format should look like the following sample.

```python
TRAIN_DATA = [('The F15 aircraft uses a lot of fuel', {'entities': [(4, 7, 'aircraft')]}),
                  ('did you see the F16 landing?', {'entities': [(16, 19, 'aircraft')]}),
                  ('how many missiles can a F35 carry', {'entities': [(24, 27, 'aircraft')]}),
                  ('is the F15 outdated', {'entities': [(7, 10, 'aircraft')]}),
                  ('does the US still train pilots to dog fight?',
                   {'entities': [(0, 0, 'aircraft')]}),
                  ('how long does it take to train a F16 pilot',
                   {'entities': [(33, 36, 'aircraft')]}),
                  ('how much does a F35 cost', {'entities': [(16, 19, 'aircraft')]}),
                  ('would it be possible to steal a F15', {'entities': [(32, 35, 'aircraft')]}),
                  ('who manufactures the F16', {'entities': [(21, 24, 'aircraft')]}),
                  ('how many countries have bought the F35',
                   {'entities': [(35, 38, 'aircraft')]}),
                  ('is the F35 a waste of money', {'entities': [(7, 10, 'aircraft')]})]
```

* Construct spaCy 3.0 training data format with given script.
* 