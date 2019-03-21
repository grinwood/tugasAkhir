from tkinter import *
import tkinter.filedialog as filedialog
import Preprocessing as prePro
import Homemade_KFOLD_sentimen as senti
import matplotlib.pyplot as plt
import numpy as np
import os

def stopProg(e):
    root.destroy()
def browsefuncpos():
    filename = filedialog.askopenfilename(filetypes=[('file teks','*.txt')])
    lpospath.config(text=filename)
def browsefuncneg():
    filename = filedialog.askopenfilename(filetypes=[('file teks','*.txt')])
    lnegpath.config(text=filename)
def getdataset():
    if (var400.get() == 1):
        prePro.new_jsonLoader(lpospath.cget('text'), lnegpath.cget('text'), 400, str(400) + 'Dataset')
    if (var600.get() == 1):
        prePro.new_jsonLoader(lpospath.cget('text'), lnegpath.cget('text'), 600, str(600) + 'Dataset')
    if (var800.get() == 1):
        prePro.new_jsonLoader(lpospath.cget('text'), lnegpath.cget('text'), 800, str(800) + 'Dataset')
    if (var1000.get() == 1):
        prePro.new_jsonLoader(lpospath.cget('text'), lnegpath.cget('text'), 1000, str(1000) + 'Dataset')
def runprepro(status):
    if (var400.get() == 1):
        prePro.runAllPreprocessing(str(400)+'Dataset',status)
    if (var600.get() == 1):
        prePro.runAllPreprocessing(str(600)+'Dataset',status)
    if (var800.get() == 1):
        prePro.runAllPreprocessing(str(800)+'Dataset',status)
    if (var1000.get()== 1):
        prePro.runAllPreprocessing(str(1000)+'Dataset',status)
#variable untuk grafik
jumDataset = [400,600,800,1000]
tff1,tfprec,tfrec = [],[],[]
deltaf1,deltaprec,deltarec = [],[],[]
def showgrap(status):
    def deltaidfbar():
        presentase = [75, 77.5, 80, 82.5, 85]
        if os.path.isfile('deltaf1.txt') == True :
            f1 = open('deltaf1.txt').read().splitlines()
            f1 = np.array(f1,float)
            prec = open('deltaprec.txt').read().splitlines()
            prec = np.array(prec,float)
            rec = open('deltarec.txt').read().splitlines()
            rec = np.array(rec,float)
            tmp = [80.29,70.1,90.66,50.5]
            N = 4
            index = np.arange(N)
            width = 0.27
            fig = plt.figure()
            ax = plt.subplot(111)

            bar1 = ax.bar(index,f1,width,color='r')
            bar2 = ax.bar(index+width,prec,width,color='g')
            bar3 = ax.bar(index+width*2,rec,width,color='b')
            ax.set_yticks(presentase)
            ax.set_yticklabels(presentase)
            ax.set_ylim(ymin=75,ymax=85)
            ax.set_ylabel('Presentase')
            ax.set_xticks(index+width)
            ax.set_xticklabels(jumDataset)

            ax.legend((bar1[0],bar2[0],bar3[0]),('F1','Precision','Recall'))
            def autolabel(rects, xpos='center'):
                """
                Attach a text label above each bar in *rects*, displaying its height.

                *xpos* indicates which side to place the text w.r.t. the center of
                the bar. It can be one of the following {'center', 'right', 'left'}.
                """

                xpos = xpos.lower()  # normalize the case of the parameter
                ha = {'center': 'center', 'right': 'left', 'left': 'right'}
                offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

                for rect in rects:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                            '{}'.format(height), ha=ha[xpos], va='bottom')

            autolabel(bar1)
            autolabel(bar2)
            autolabel(bar3)
            plt.title('Delta TF-IDF')
            plt.show()
    def tfidfbar():
        presentase = [72.5,75, 77.5, 80, 82.5, 85]
        if os.path.isfile('tff1.txt') == True :
            f1 = open('tff1.txt').read().splitlines()
            f1 = np.array(f1,float)
            prec = open('tfprec.txt').read().splitlines()
            prec = np.array(prec,float)
            rec = open('tfrec.txt').read().splitlines()
            rec = np.array(rec,float)
            tmp = [80.29,70.1,90.66,50.5]
            N = 4
            index = np.arange(N)
            width = 0.27
            fig = plt.figure()
            ax = plt.subplot(111)

            bar1 = ax.bar(index,f1,width,color='r')
            bar2 = ax.bar(index+width,prec,width,color='g')
            bar3 = ax.bar(index+width*2,rec,width,color='b')
            ax.set_yticks(presentase)
            ax.set_yticklabels(presentase)
            ax.set_ylim(ymin=72.5,ymax=85)
            ax.set_ylabel('Presentase')
            ax.set_xticks(index+width)
            ax.set_xticklabels(jumDataset)

            ax.legend((bar1[0],bar2[0],bar3[0]),('F1','Precision','Recall'))
            def autolabel(rects, xpos='center'):
                """
                Attach a text label above each bar in *rects*, displaying its height.

                *xpos* indicates which side to place the text w.r.t. the center of
                the bar. It can be one of the following {'center', 'right', 'left'}.
                """

                xpos = xpos.lower()  # normalize the case of the parameter
                ha = {'center': 'center', 'right': 'left', 'left': 'right'}
                offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

                for rect in rects:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                            '{}'.format(height), ha=ha[xpos], va='bottom')

            autolabel(bar1)
            autolabel(bar2)
            autolabel(bar3)
            plt.title('TF-IDF')
            plt.show()
    def komparasi():
        presentase = [72.5, 75, 77.5, 80, 82.5, 85]
        if os.path.isfile('tff1.txt') == True:
            f1 = open('tff1.txt').read().splitlines()
            f1 = np.array(f1, float)
            sc2f1 = open('sc2tff1.txt').read().splitlines()
            sc2f1 = np.array(sc2f1,float)
            delf1 = open('deltaf1.txt').read().splitlines()
            delf1 = np.array(delf1,float)
            sc2delf1 = open('sc2deltaf1.txt').read().splitlines()
            sc2delf1 = np.array(sc2delf1,float)
            N = 4
            jumbar = 4
            offset = 0
            index = np.arange(N)
            width = 1/(jumbar + 2)
            fig = plt.figure()
            ax = plt.subplot(111)

            bar1 = ax.bar(index, f1, width, color='r')
            bar2 = ax.bar(index + width, sc2f1, width, color='b')
            bar3 = ax.bar(index + width*2, delf1, width, color='orange')
            bar4 = ax.bar(index + width*3,sc2delf1,width,color='g',)
            ax.set_yticks(presentase)
            ax.set_yticklabels(presentase)
            ax.set_ylim(ymin=72.5, ymax=85)
            ax.set_ylabel('Presentase')
            ax.set_xticks(index + (len(f1)/2)*width)
            ax.set_xticklabels(jumDataset)

            ax.legend((bar1[0], bar2[0],bar3[0],bar4[0]), ('F1', 'F1 setelah menggunakan FS','delta F1','delta F1 setelah FS'))

            def autolabel(rects, xpos='center'):
                """
                Attach a text label above each bar in *rects*, displaying its height.

                *xpos* indicates which side to place the text w.r.t. the center of
                the bar. It can be one of the following {'center', 'right', 'left'}.
                """

                xpos = xpos.lower()  # normalize the case of the parameter
                ha = {'center': 'center', 'right': 'left', 'left': 'right'}
                offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

                for rect in rects:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                            '{}'.format(height), ha=ha[xpos], va='bottom')

            autolabel(bar1)
            autolabel(bar2)
            autolabel(bar3)
            autolabel(bar4)
            plt.title('Komparasi TF-IDF')
            plt.show()
    if status =='delta':
        deltaidfbar()
    if status=='tf':
        tfidfbar()
    if status=='komtff1':
        komparasi()
#showgrap('tf')
def runtfidf():
    def input(f1,prec,rec):
        file = open('tff1.txt', 'a')
        file.write(str(f1) + '\n')
        file.close()
        file = open('tfprec.txt', 'a')
        file.write(str(prec) + '\n')
        file.close()
        file = open('tfrec.txt', 'a')
        file.write(str(rec) + '\n')
        file.close()
    if (var400.get() == 1):
        f1, prec, rec = senti.fold_tfidf(str(400)+ 'Dataset')
        input(round((f1 * 100), 3), round((prec * 100), 3), round((rec * 100), 3))
    if (var600.get() == 1):
        f1, prec, rec = senti.fold_tfidf(str(600)+'Dataset')
        input(round((f1 * 100), 3), round((prec * 100), 3), round((rec * 100), 3))
    if (var800.get() == 1):
        f1, prec, rec = senti.fold_tfidf(str(800)+'Dataset')
        input(round((f1 * 100), 3), round((prec * 100), 3), round((rec * 100), 3))
    if (var1000.get() == 1):
        f1,prec,rec = senti.fold_tfidf(str(1000)+'Dataset')
        input(round((f1 * 100), 3), round((prec * 100), 3), round((rec * 100), 3))
def rundelta():
    def input(f1,prec,rec):
        file = open('deltaf1.txt', 'a')
        file.write(str(f1) + '\n')
        file.close()
        file = open('deltaprec.txt', 'a')
        file.write(str(prec) + '\n')
        file.close()
        file = open('deltarec.txt', 'a')
        file.write(str(rec) + '\n')
        file.close()
    if (var400.get() == 1):
        f1,prec,rec = senti.fold_deltatfidf(str(400)+'Dataset')
        input(round((f1*100),3),round((prec*100),3),round((rec*100),3))
    if (var600.get() == 1):
        f1, prec, rec = senti.fold_deltatfidf(str(600)+'Dataset')
        input(round((f1 * 100), 3), round((prec * 100), 3), round((rec * 100), 3))
    if (var800.get() == 1):
        f1, prec, rec = senti.fold_deltatfidf(str(800)+'Dataset')
        input(round((f1 * 100), 3), round((prec * 100), 3), round((rec * 100), 3))
    if (var1000.get() == 1):
        f1, prec, rec = senti.fold_deltatfidf(str(1000)+'Dataset')
        input(round((f1 * 100), 3), round((prec * 100), 3), round((rec * 100), 3))
root = Tk()
#root.geometry('800x400')


# create all of the main containers
fbrowse = Frame(root,bg='cyan',width=450, height=50)
fjumdata = Frame(root,bg='yellow',width=450, height=100)
faction = Frame(root,bg='red',width=450, height=100)
freport = Frame(root,bg='green',width=450, height=100)

# layout all of the main containers
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

fbrowse.grid(row=0,sticky='w')
fjumdata.grid(row=1,sticky='nw')
faction.grid(row=2,sticky='w')
freport.grid(row=3,sticky='w')
# Fbrowser frame
labjudul = StringVar()
labjudul.set('TA Rian')
ljudul = Label(fbrowse, textvariable=labjudul)
ljudul.grid(row=0,column=1)
lposdataset = Label(fbrowse, text='Positif dataset   : ').grid(row=1,column=1)
lpospath = Label(fbrowse,text='none')
lpospath.grid(row=1,column=3)
butposbrowse = Button(fbrowse, text="Browse", command=lambda :browsefuncpos()).grid(row=1,column=2)
lnegdataset = Label(fbrowse, text='Negatif dataset : ').grid(row=2,column=1)
lnegpath = Label(fbrowse,text='none')
lnegpath.grid(row=2,column=3)
butnegbrowse = Button(fbrowse, text="Browse", command=lambda :browsefuncneg()).grid(row=2,column=2)

# fjumdata frame
ljumdata = Label(fjumdata,text='jumlah data : ').grid(row=0,column=1)

var400=IntVar()
cek400= Checkbutton(fjumdata,text='400',variable=var400).grid(row=0,column=2)
var600=IntVar()
cek600= Checkbutton(fjumdata,text='600',variable=var600).grid(row=0,column=3)
var800=IntVar()
cek800= Checkbutton(fjumdata,text='800',variable=var800).grid(row=0,column=4)
var1000=IntVar()
cek1000 = Checkbutton(fjumdata,text='1000',variable=var1000).grid(row=0,column=5)

# faction frame
laction = Label(faction,text='Action').grid(row=0,columnspan=5)
butgetdataset = Button(faction,text='Ambil dataset',activebackground='yellow',command=lambda :getdataset()).grid(row=1,column=1)
butpreprosc1 = Button(faction,text='Prepro Sce 1 (w/out seleksi fitur)',activebackground='yellow',command=lambda :runprepro('off')).grid(row=1,column=2)
butpreprosc2 = Button(faction,text='Prepro Sce 2 (w/ seleksi fitur)',activebackground='yellow',command=lambda :runprepro('on')).grid(row=1,column=3)
buttfidf = Button(faction,text='Analisis sentimen + pembobotan TF-IDF',activebackground='yellow',command=lambda :runtfidf()).grid(row=1,column=4)
butdelta = Button(faction,text='Analisis sentimen + pembobotan Delta TF-IDF',activebackground='yellow',command=lambda :rundelta()).grid(row=1,column=5)

# freport frame
lreport = Label(freport,text='Report').grid(row=0,columnspan=5)
butrptdelta =  Button(freport,text='Show report Delta',activebackground='yellow',command=lambda :showgrap('delta')).grid(row=1,column=0)
butrpttf = Button(freport,text='show report TF',activebackground='yellow',command=lambda :showgrap('tf')).grid(row=1,column=1)
butkomtf = Button(freport,text='show report komparasi TF F1',activebackground='yellow',command=lambda :showgrap('komtff1')).grid(row=1,column=2)
root.mainloop()