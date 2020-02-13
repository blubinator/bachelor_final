import os 
  
def main(): 
    i = 0
      
    for filename in os.listdir("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures"): 
        dst ="fuehrerschein" + str(i) + ".jpg"
        src = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures\\" + filename  
        dst = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures\\" + dst  
        os.rename(src, dst) 
        i += 1
  

if __name__ == '__main__': 
    main() 