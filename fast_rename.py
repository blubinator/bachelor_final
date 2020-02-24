import os 
  
def main(): 
    i = 0
      
    for filename in os.listdir("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures_idcard"): 
        dst ="ausweis" + str(i) + ".jpg"
        src = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures_idcard\\" + filename  
        dst = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures_idcard\\" + dst  
        os.rename(src, dst) 
        i += 1
  

if __name__ == '__main__': 
    main() 