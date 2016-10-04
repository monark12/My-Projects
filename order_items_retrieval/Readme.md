# The Project Is an example of 'Text Mining'.
# Text mining, also referred to as text data mining, roughly equivalent to text analytics, refers to the process of deriving high-quality information from text

The problem is to extracts item name, quantity, unit and special comments from a sentence of orders.
The data we used is of NYTimes tagged by human news assistants

For example-
Suppose a person enters the following text(i.e he wants)

2 1/4 cups all-purpose flour.

We have to extract the informations about what the person wants and how much in a particulat format.

[
'name': 'all-purpose flour',

'qty': '2$1/4',

'unit': 'cup'
  
]

For this we are using Python-crfsuite module to create and train our model.  
