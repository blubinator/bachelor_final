import os 
  
def main(): 
    i = 0
      
    for filename in os.listdir("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures_idcard\\front"): 
        dst ="ausweis" + str(i) + ".png"
        src = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures_idcard\\front\\" + filename  
        dst = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures_idcard\\front\\" + dst  
        os.rename(src, dst) 
        i += 1
  

if __name__ == '__main__': 
    main() 