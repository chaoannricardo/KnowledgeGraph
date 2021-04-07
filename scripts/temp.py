if __name__ == '__main__':
    file = open("./stopwords/numbers.txt", mode="w")
    for index in range(-100000, 100000):
        file.write(str(index) + "\n")

file.close()