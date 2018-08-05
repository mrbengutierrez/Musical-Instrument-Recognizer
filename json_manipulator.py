"""
This file shows how to manipulate json data
"""

import json


def genList():
    a = []
    n = 5
    m = 4
    l = 3
    for i in range(n):
        a.append([])
        for j in range(m):
            a[i].append([])
            for k in range (l):
                a[i][j].append( (i,j,k))
    return a

def doJsonStuff():
    data = {}
    data['weights'] = genList()
    print('After Opening File')
    print('type(data[\'weights\']) = ' + str(type(data['weights'])))
    print('data[\'weights\'] = ' + str(data['weights']))
    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile)

    data = 0    
    with open('data.txt') as json_file:  
        data = json.load(json_file)
        print('---')
        print('After Opening File')
        print('type(data[\'weights\']) = ' + str(type(data['weights'])))
        print('data[\'weights\'] = ' + str(data['weights']))
        






def main():
    doJsonStuff()


if __name__ == "__main__":
    main()
