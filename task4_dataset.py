import pandas as pd

# Sample dataset
data = {
    'text': [
        'Free entry in 2 a weekly competition!',
        'Hey, are we meeting today?',
        'Congratulations! You have won a prize.',
        'Please call me back.',
        'Win a $1000 gift card now!',
        'Can we have lunch tomorrow?'
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)
print(df)
