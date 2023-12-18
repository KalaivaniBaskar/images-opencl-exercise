#include <iostream>
#include <vector>
using namespace std;

int main() 
{   
   vector<vector<int> > arr = { {1,2,3,12} , {4,5,6,18} , {7,8,9,21} };
    cout << "Hello, World!";
    cout << arr[0][2] << endl;
    int rows = 3;
    int cols = 4; 
    vector < vector<int> > op;
    for(int i = 0 ; i < rows; i++){
      for(int j = 0 ; j < (cols/2); j++){
         int temp = arr[i][j];
         arr[i][j] = arr[i][cols - j - 1];
         arr[i][cols - j - 1] = temp ;
         
      }
    } 
    
    cout << arr[0][0] << endl;
    cout << arr[0][1] << endl;
    cout << arr[0][2] << endl;
    cout << arr[0][3] << endl;
    return 0;
}