#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>

#include <json.hpp>
using json = nlohmann::json;

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @json j        |        | path to json }"
        ;


int main(int argc, char** argv)
{
    /* Parse command-line args */
    cv::CommandLineParser parser(argc, argv, argKeys);
    parser.about("JSON parser");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string jsonPath = parser.get<std::string>("@json");
    if ( jsonPath.empty() ) 
    {
        std::cerr << "[ERROR] JSON path must not be empty" << std::endl;
        return 0;
    }
    
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    std::cout << ">>> JSON file: " << jsonPath << std::endl;

    std::ifstream i(jsonPath.c_str());
    if ( !i.good() )
    {
        std::cerr << "[ERROR] Could not read JSON file. Possibly file does not exist" << std::endl;
        return 0;
    }

    json j;
    i >> j;
    if ( j.empty() )
    {
        std::cerr << "[ERROR] Could not create JSON object from file" << std::endl;
        return 0;
    }

    std::cout << ">>> JSON: \n" << j.dump(4) << std::endl;

    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
