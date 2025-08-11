#include<iostream>
#include<fstream>

#include "Vector.hpp"
#include "Color.hpp"
#include "Point.hpp"
#include "Ray.hpp"

using namespace std;

const int IMG_HEIGHT = 512;
const int IMG_WIDTH = 512;

void write_image(const int IMG_HEIGHT, const int IMG_WIDTH, ostream& cout) {
    // write ppm headers
    cout << "P3\n" << IMG_WIDTH << " " << IMG_HEIGHT << "\n255\n";
    
    for(int j = 0; j < IMG_HEIGHT; j++){
        for(int i = 0; i < IMG_WIDTH; i++){

            float r = float(i) / float(IMG_HEIGHT-1);
            float g =  float(j) / float(IMG_WIDTH-1);
            float b = 0.0f;

            // write ppm data
            Color c(r, g, b); // [0-1] range
            write_color(cout, c);

        }
    }
}

int main() {
    // Create a vector
    Vector v(3.0, 4.0, 5.0);
    
    // Output the vector
    cout << "Vector: " << v << endl;

    // Get the length of the vector
    cout << "Length: " << v.length() << endl;

    // Normalize the vector
    v.make_unit_vector();
    cout << "Normalized Vector: " << v << endl;

    // Output the squared length
    cout << "Squared Length: " << v.squared_length() << endl;

    // Write an image to stdout
    // write_image(IMG_HEIGHT, IMG_WIDTH, std::cout);

    // write to a file
    ofstream out("output.ppm");
    if (!out) {
        cerr << "Error opening file for writing." << endl;
        return 1;
    }
    // Write the image to the file
    write_image(IMG_HEIGHT, IMG_WIDTH, out);

    out.close();
    cout << "Image written to output.ppm" << endl;

    // Example of using Ray
    Point origin(1, 1, 1);
    Vector direction(1, 0, 0);
    Ray ray(origin, direction);
    cout << "Ray: " << ray << endl;
    cout << "Ray at t=2: " << ray.at(2) << endl;
    cout << "Ray scaled by 2 at 5: " << ray.scaled(2).at(5) << endl;
    


    return 0;
}