def tokenizerx(data):
    invalobj = [
        '!', '@', '#', '$', '%', '^', '&', '*',
        '(', ')', "[", "]", "{", "}", "\\", '-',
        '=', '_', '+', '`', '~', ';', ':', "'",
        ',', '<', '.', '>', '?', '/', '|', '"'
    ]
    for i in range(len(invalobj)):
        if invalobj[i] in data:
            data = "".join(data.split(invalobj[i]))

    return data
