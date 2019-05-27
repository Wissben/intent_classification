import json


def parse_ask_ubuntu():
    ret = []
    fd = open('AskUbuntuCorpus.json', 'r')
    data = json.load(fd)
    for sent in data['sentences']:
        ret.append({
            "id": 1,
            "text": sent['text'],
            "intent": "unknown"
        })
    return ret


def parse_chatbots():
    ret = []
    fd = open('ChatbotCorpus.json', 'r')
    data = json.load(fd)
    for sent in data['sentences']:
        ret.append({
            "id": 1,
            "text": sent['text'],
            "intent": "unknown"
        })
    return ret


def parse_benchmark():
    ret = []
    fd = open('benchmark_data.json', 'r')
    data = json.load(fd)
    for ints in data['domains']:
        for intent in ints['intents']:
            for query in intent['queries']:
                ret.append({
                    "id": 1,
                    "text": query['text'],
                    "intent": "unknown"
                })
    return ret


ret = parse_ask_ubuntu()
ret += parse_chatbots()
ret += parse_benchmark()

json.dump(ret, open('out.json', 'w+'))
