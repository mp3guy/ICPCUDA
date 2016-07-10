#include "ICPOdometry.h"

#include <iomanip>
#include <fstream>

std::ifstream asFile;
std::string directory;

void tokenize(const std::string & str, std::vector<std::string> & tokens, std::string delimiters = " ")
{
    tokens.clear();

    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
}

uint64_t loadDepth(cv::Mat1w & depth)
{
    std::string currentLine;
    std::vector<std::string> tokens;
    std::vector<std::string> timeTokens;

    do
    {
        getline(asFile, currentLine);
        tokenize(currentLine, tokens);
    } while(tokens.size() > 2);

    if(tokens.size() == 0)
        return 0;

    std::string depthLoc = directory;
    depthLoc.append(tokens[1]);
    depth = cv::imread(depthLoc, CV_LOAD_IMAGE_ANYDEPTH);

    tokenize(tokens[0], timeTokens, ".");

    std::string timeString = timeTokens[0];
    timeString.append(timeTokens[1]);

    uint64_t time;
    std::istringstream(timeString) >> time;

    for(unsigned int i = 0; i < 480; i++)
    {
        for(unsigned int j = 0; j < 640; j++)
        {
            depth.at<unsigned short>(i, j) /= 5;
        }
    }

    return time;
}

void outputFreiburg(const std::string filename, const int64_t & timestamp, const Eigen::Matrix4f & currentPose)
{
    std::ofstream file;
    file.open(filename.c_str(), std::fstream::app);

    std::stringstream strs;

    strs << std::setprecision(6) << std::fixed << (double)timestamp / 1000000.0 << " ";

    Eigen::Vector3f trans = currentPose.topRightCorner(3, 1);
    Eigen::Matrix3f rot = currentPose.topLeftCorner(3, 3);

    file << strs.str() << trans(0) << " " << trans(1) << " " << trans(2) << " ";

    Eigen::Quaternionf currentCameraRotation(rot);

    file << currentCameraRotation.x() << " " << currentCameraRotation.y() << " " << currentCameraRotation.z() << " " << currentCameraRotation.w() << "\n";

    file.close();
}

int main(int argc, char * argv[])
{
    assert((argc == 2 || argc == 3) && "Please supply the depth.txt dir as the first argument");

    directory.append(argv[1]);

    if(directory.at(directory.size() - 1) != '/')
    {
        directory.append("/");
    }

    std::string associationFile = directory;
    associationFile.append("depth.txt");

    asFile.open(associationFile.c_str());

    cv::Mat1w firstRaw(480, 640);
    cv::Mat1w secondRaw(480, 640);

    ICPOdometry icpOdom(640, 480, 320, 240, 528, 528);

    assert(!asFile.eof() && asFile.is_open());

    loadDepth(firstRaw);
    uint64_t timestamp = loadDepth(secondRaw);

    Eigen::Matrix4f currPose = Eigen::Matrix4f::Identity();

    std::ofstream file;
    file.open("output.poses", std::fstream::out);
    file.close();

    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);

    std::string dev(prop.name);

    std::cout << dev << std::endl;

    float mean = std::numeric_limits<float>::max();
    int count = 0;

    int threads = 128;
    int blocks = 96;

    int bestThreads = threads;
    int bestBlocks = blocks;
    float best = mean;

    if(argc == 3)
    {
        std::string searchArg(argv[2]);

        if(searchArg.compare("-v") == 0)
        {
            std::cout << "Searching for the best thread/block configuration for your GPU..." << std::endl;
            std::cout << "Best: " << bestThreads << " threads, " << bestBlocks << " blocks (" << best << "ms)"; std::cout.flush();

            float counter = 0;

            for(threads = 16; threads <= 512; threads += 16)
            {
                for(blocks = 16; blocks <= 512; blocks += 16)
                {
                    mean = 0.0f;
                    count = 0;

                    for(int i = 0; i < 5; i++)
                    {
                        icpOdom.initICPModel((unsigned short *)firstRaw.data, 20.0f, currPose);
                        icpOdom.initICP((unsigned short *)secondRaw.data, 20.0f);

                        Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
                        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);

                        boost::posix_time::ptime time = boost::posix_time::microsec_clock::local_time();
                        boost::posix_time::time_duration duration1(time.time_of_day());
                        unsigned long long int tick = duration1.total_microseconds();

                        icpOdom.getIncrementalTransformation(trans, rot, threads, blocks);

                        time = boost::posix_time::microsec_clock::local_time();
                        boost::posix_time::time_duration duration2(time.time_of_day());
                        unsigned long long int tock = duration2.total_microseconds();

                        mean = (float(count) * mean + (tock - tick) / 1000.0f) / float(count + 1);
                        count++;
                    }

                    counter++;

                    if(mean < best)
                    {
                        best = mean;
                        bestThreads = threads;
                        bestBlocks = blocks;
                    }

                    std::cout << "\rBest: " << bestThreads << " threads, " << bestBlocks << " blocks (" << best << "ms), " << int((counter / 1024.f) * 100.f) << "%    "; std::cout.flush();
                }
            }

            std::cout << std::endl;
        }
    }

    threads = bestThreads;
    blocks = bestBlocks;

    mean = 0.0f;
    count = 0;

    while(!asFile.eof())
    {
        icpOdom.initICPModel((unsigned short *)firstRaw.data, 20.0f, currPose);

        icpOdom.initICP((unsigned short *)secondRaw.data, 20.0f);

        Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);

        boost::posix_time::ptime time = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration duration1(time.time_of_day());
        unsigned long long int tick = duration1.total_microseconds();

        icpOdom.getIncrementalTransformation(trans, rot, threads, blocks);

        time = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration duration2(time.time_of_day());
        unsigned long long int tock = duration2.total_microseconds();

        currPose.topLeftCorner(3, 3) = rot;
        currPose.topRightCorner(3, 1) = trans;


        mean = (float(count) * mean + (tock - tick) / 1000.0f) / float(count + 1);
        count++;

        std::cout << std::setprecision(4) << std::fixed
                  << "\r ICP: "
                  << mean;
                  std::cout.flush();

        std::swap(firstRaw, secondRaw);

        outputFreiburg("output.poses", timestamp, currPose);

        timestamp = loadDepth(secondRaw);
    }

    std::cout << std::endl;

    std::cout << "ICP speed: " << int(1000.f / mean) << "Hz" << std::endl;

    return 0;
}

