"""
DESCRIPTION:
This file takes instrumental data in one directory, and organizes it by notes
in new directory

Note: Notes of all octaves are stored in a single note folder.
        I.E. A2, A3, A4, A5 => A


MIT License

Copyright (c) 2018 The-Instrumental-Specialists

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import glob
import os
import shutil

def createFolder(directory):
    """Creates a folder in directory. Throws Error if folder exists"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        raise OSError


        
def note_sort(directory,new_dir):
    """Takes instrumental data in directory and organizes it by notes in new directory""""
    if os.path.isdir(new_dir):
        raise OSError('Note directory already exists, please delete or rename')
        
    dirs = [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]
    files = []

    for d in dirs: 
        sub_dir = os.path.join(directory, d)
        for filename in glob.glob(os.path.join(sub_dir, '*.wav')):
            files.append(filename)
            
    notes = {'A':[],'B':[],'C':[],'D':[],'E':[],'F':[],'G':[],
             'Fs':[],'Cs':[],'Gs':[],'Ds':[],'As':[]}
    octaves = ['1','2','3','4','5','6','7','8','9']
    for file in files:
        for note in notes:
            for octave in octaves:
                word = '_' + note + octave + '_'
                if word in file:
                    notes[note].append(file)
                    
    os.makedirs(new_dir)
    for note in notes:
        sub_dir = new_dir + '/' + note
        os.makedirs(sub_dir)
        for file in notes[note]:
            shutil.copy2(file, sub_dir)

    
            
    
                

    



def main():
    note_sort('phil_temp_02','cat')







if __name__ == '__main__':
    main()
