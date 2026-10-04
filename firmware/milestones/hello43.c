#include <stdio.h>
int main(int argc,char**argv){
  printf("TEAM43_STANDALONE_OK argc=%d\n",argc);
  if(argc>1) printf("arg1=%s\n",argv[1]);
  return 0;
}
