if __name__ == '__main__':
    file = open("./stopwords/numbers.txt", mode="w")
    for index in range(-1000000, 1000000):
        file.write(str(index) + "\n")

file.close()