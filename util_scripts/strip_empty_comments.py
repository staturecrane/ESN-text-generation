with open('parsed_python.txt', 'r') as input_file:
    lines = input_file.readlines()
    with open('stripped_python.txt', 'w') as output_file:
        for line in lines:
            if line.strip() not in ['##', '#--']:
                output_file.write(line)
