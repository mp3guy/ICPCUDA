#include "ICPOdometry.h"
#include "ICPSlowdometry.h"

#include <iomanip>
#include <fstream>
#include <chrono>

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

    getline(asFile, currentLine);
    tokenize(currentLine, tokens);

    if(tokens.size() == 0)
        return 0;

    std::string depthLoc = directory;
    depthLoc.append(tokens[3]);
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
    namespace ch = std::chrono;

    assert((argc == 2 || argc == 3) && "Please supply the association file dir as the first argument");

    directory.append(argv[1]);

    if(directory.at(directory.size() - 1) != '/')
    {
        directory.append("/");
    }

    std::string associationFile = directory;
    associationFile.append("association.txt");

    asFile.open(associationFile.c_str());

    cv::Mat1w firstRaw(480, 640);
    cv::Mat1w secondRaw(480, 640);

    ICPOdometry icpOdom(640, 480, 320, 240, 528, 528);
    ICPSlowdometry icpSlowdom(640, 480, 320, 240, 528, 528);

    assert(!asFile.eof() && asFile.is_open());

    loadDepth(firstRaw);
    uint64_t timestamp = loadDepth(secondRaw);

    Eigen::Matrix4f currPoseFast = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f currPoseSlow = Eigen::Matrix4f::Identity();

    std::ofstream file;
    file.open("fast.poses", std::fstream::out);
    file.close();

    file.open("slow.poses", std::fstream::out);
    file.close();

    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);

    std::string dev(prop.name);

    std::cout << dev << std::endl;

    float meanFast = std::numeric_limits<float>::max();
    float meanSlow = 0.0f;
    int count = 0;

    int threads = 128;
    int blocks = 96;

    int bestThreads = threads;
    int bestBlocks = blocks;
    float bestFast = meanFast;

    if(argc == 3)
    {
        std::string searchArg(argv[2]);

        if(searchArg.compare("-v") == 0)
        {
            std::cout << "Searching for the best thread/block configuration for your GPU..." << std::endl;
            std::cout << "Best: " << bestThreads << " threads, " << bestBlocks << " blocks (" << bestFast << "ms)"; std::cout.flush();

            float counter = 0;

            for(threads = 16; threads <= 512; threads += 16)
            {
                for(blocks = 16; blocks <= 512; blocks += 16)
                {
                    meanFast = 0.0f;
                    count = 0;

                    for(int i = 0; i < 5; i++)
                    {
                        icpOdom.initICPModel((unsigned short *)firstRaw.data, 20.0f, currPoseFast);
                        icpOdom.initICP((unsigned short *)secondRaw.data, 20.0f);

                        Eigen::Vector3f trans = currPoseFast.topRightCorner(3, 1);
                        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPoseFast.topLeftCorner(3, 3);

                        auto tick = ch::high_resolution_clock::now();
                        icpOdom.getIncrementalTransformation(trans, rot, threads, blocks);
                        auto tock = ch::high_resolution_clock::now();

                        meanFast = (float(count) * meanFast + ch::duration_cast< ch::milliseconds >(tock-tick).count()) / float(count + 1);
                        count++;
                    }

                    counter++;

                    if(meanFast < bestFast)
                    {
                        bestFast = meanFast;
                        bestThreads = threads;
                        bestBlocks = blocks;
                    }

                    std::cout << "\rBest: " << bestThreads << " threads, " << bestBlocks << " blocks (" << bestFast << "ms), " << int((counter / 1024.f) * 100.f) << "%    "; std::cout.flush();
                }
            }

            std::cout << std::endl;
        }
    }

    threads = bestThreads;
    blocks = bestBlocks;

    meanFast = 0.0f;
    meanSlow = 0.0f;
    count = 0;

    while(!asFile.eof())
    {
        icpOdom.initICPModel((unsigned short *)firstRaw.data, 20.0f, currPoseFast);

        icpOdom.initICP((unsigned short *)secondRaw.data, 20.0f);

        Eigen::Vector3f trans = currPoseFast.topRightCorner(3, 1);
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPoseFast.topLeftCorner(3, 3);

        auto tick_fast = ch::high_resolution_clock::now();
        icpOdom.getIncrementalTransformation(trans, rot, threads, blocks);
        auto tock_fast = ch::high_resolution_clock::now();

        currPoseFast.topLeftCorner(3, 3) = rot;
        currPoseFast.topRightCorner(3, 1) = trans;

        icpSlowdom.initICPModel((unsigned short *)firstRaw.data, 20.0f, currPoseSlow);

        icpSlowdom.initICP((unsigned short *)secondRaw.data, 20.0f);

        trans = currPoseSlow.topRightCorner(3, 1);
        rot = currPoseSlow.topLeftCorner(3, 3);

        auto tick_slow = ch::high_resolution_clock::now();
        icpSlowdom.getIncrementalTransformation(trans, rot);
        auto tock_slow = ch::high_resolution_clock::now();

        currPoseSlow.topLeftCorner(3, 3) = rot;
        currPoseSlow.topRightCorner(3, 1) = trans;

        meanFast = (float(count) * meanFast + ch::duration_cast< ch::milliseconds >(tock_fast - tick_fast).count()) / float(count + 1);
        meanSlow = (float(count) * meanSlow + ch::duration_cast< ch::milliseconds >(tock_slow - tick_slow).count()) / float(count + 1);
        count++;

        std::cout << std::setprecision(4) << std::fixed
                  << "\rFast ICP: "
                  << meanFast
                  << "ms, Slow ICP: "
                  << meanSlow << "ms";
                  std::cout.flush();

        std::swap(firstRaw, secondRaw);

        outputFreiburg("fast.poses", timestamp, currPoseFast);
        outputFreiburg("slow.poses", timestamp, currPoseSlow);

        timestamp = loadDepth(secondRaw);
    }

    std::cout << std::endl;

    std::cout << meanSlow / meanFast << " times faster." << std::endl;

    return 0;
}

