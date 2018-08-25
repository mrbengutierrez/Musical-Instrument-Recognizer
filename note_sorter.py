import glob
import os
import shutil

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        raise OSError


        
def note_sort(directory,new_dir):
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
