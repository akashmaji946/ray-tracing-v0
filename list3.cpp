#include <iostream>
using namespace std;

const int IMG_HEIGHT = 1024;
const int IMG_WIDTH = 1024;

int main(){

    // write ppm headers
    cout << "P3\n" << IMG_WIDTH << " " << IMG_HEIGHT << "\n255\n";
    
    for(int j = 0; j < IMG_HEIGHT; j++){
        // logging for progress using `clog`
        clog << "\rLeft lines: " << (IMG_HEIGHT-j) << " " << flush;
        for(int i = 0; i < IMG_WIDTH; i++){
            float r = float(i) / float(IMG_HEIGHT-1);
            float g =  float(j) / float(IMG_WIDTH-1);
            float b = 0.0f;

            int rr = int(r * 255);
            int gg = int(g * 255);
            int bb = int(b * 255);

            // write ppm data
            cout << rr << " " << gg << " " << bb << "\n";

        }
    }
    clog << "\rDone.\n";
    return 0;

}