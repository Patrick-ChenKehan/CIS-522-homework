with open("./model.py", "r") as f:
    text = f.readlines()
    text.insert(31, "# Comment")

with open("./model.py", "w") as f:
    f.writelines(text)
