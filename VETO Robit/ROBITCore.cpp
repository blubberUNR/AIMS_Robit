// 100% completely authored by David McFadden

//System includes
#include <string>
#include <iostream>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>

//CrossSock includes
#include "CrossClient.h"
#include "CrossServer.h"
using namespace CrossSock;

//Stereo (flycap & triclops) includes
#include "triclops.h"
#include "fc2triclops.h"

//Aria includes
#include "Aria.h"

//Arnl includes
#include "Arnl.h"
#include "ArLocalizationTask.h"
#include "ArDocking.h"
#include "ArSystemStatus.h"

//OpenCV includes
#include "opencv.hpp"
#include <highgui.h>

// namespaces and aliases
namespace FC2 = FlyCapture2;
namespace FC2T = Fc2Triclops;
using namespace cv;
using namespace std;

#define PTZ_SERIAL_PORT ArUtil::COM2

#define RESIZE_HEIGHT 240//120 //240//480//192
#define RESIZE_WIDTH 320//160 //320//640//256
#define IMAGE_QUALITY 30
#define CHUNK_SIZE 1024//4800 //$$$ 1024
#define WHEEL_TIMEOUT 400
#define LRF_SEGMENT 10
#define BUMPER_BROADCAST_DELAY 200
#define LRF_BROADCAST_DELAY 250
#define SONAR_BROADCAST_DELAY 250
#define IMAGE_CHUNK_BROADCAST_DELAY 50 // 5 //$$$ 50
#define POSITION_BROADCAST_DELAY 5 // 50 // $$$ 5
#define FACING_BROADCAST_DELAY 100
#define MAX_PACKET_SIZE 1024
#define PI 3.14159265

bool inRange(double min, double max, double threshold)
{
  return (abs(max-min) <= threshold);
}

bool inRangeAngle(double min, double max, double threshold)
{
	if(inRange(min + 360, max, threshold))
		return true;
	if(inRange(min, max + 360, threshold))
		return true;
	return inRange(min, max, threshold);
}

double distanceFormula(double x1, double x2, double y1, double y2)
{
  return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
}

// struct containing image needed for processing
struct ImageContainer
{
	FC2::Image tmp[2];
	FC2::Image unprocessed[2];
	FC2::Image bgru[2];
	FC2::Image packed;
} ;

// enum to specify image side
enum IMAGE_SIDE
{
	RIGHT = 0, LEFT
};

// BOO ON POINT GREY CAMERAS THAT TAKE 250 MS A SWIG
struct  StereoImage
{
	bool shouldQuit;
	bool hasNewImage;
	vector<uchar> leftBuf;
	vector<uchar> rightBuf;

	StereoImage()
	{
		shouldQuit = false;
		hasNewImage = false;
	}
};


// Sonar reading to broadcast
struct RobotSonar
{
public:
	double originX;
	double originY;
	double originZ;
	double originPitch;
	double originYaw;
	double originRoll;
	double maxRange;
	int sonarID;
	bool isValid; // if our last reading was valid
	float lastReading;

	RobotSonar()
	{
		originX = 0.0;
		originY = 0.0;
		originZ = 0.0;
		originPitch = 0.0;
		originYaw = 0.0;
		originRoll = 0.0;
		maxRange = 0.0;
		sonarID = 0;
		isValid = false;
		lastReading = 0.0f;
	}

	RobotSonar(double x, double y, double z, double pitch, double yaw, double roll, double range, int id)
	{
		originX = x;
		originY = y;
		originZ = z;
		originPitch = pitch;
		originYaw = yaw;
		originRoll = roll;
		maxRange = range;
		sonarID = id;
		isValid = false;
		lastReading = 0.0f;
	}
};

// Camera hardpoints
struct RobotCamera
{
public:
	double originX;
	double originY;
	double originZ;
	double originPitch;
	double originYaw;
	double originRoll;
	int cameraIndex;

	RobotCamera()
	{
		originX = 0.0;
		originY = 0.0;
		originZ = 0.0;
		originPitch = 0.0;
		originYaw = 0.0;
		originRoll = 0.0;
		cameraIndex = 0;
	}

	RobotCamera(double x, double y, double z, double pitch, double yaw, double roll, int index)
	{
		originX = x;
		originY = y;
		originZ = z;
		originPitch = pitch;
		originYaw = yaw;
		originRoll = roll;
		cameraIndex = 0;
	}
};

enum ESoundToPlay {
	None,
	Connection,
	Disconnection,
	Reconnection,
	EmergencyStop,
	PathingStart,
	Goal,
	SendHome
};

//Prototypes
int initializeStereo(FC2::Camera & camera, TriclopsContext & triclops);
int disconnectStereo(FC2::Camera & camera, TriclopsContext & triclops);

void readConfigFile();

int configureCamera( FC2::Camera &camera );
int generateTriclopsContext( FC2::Camera & camera, TriclopsContext & triclops );
int grabImage (  FC2::Camera & camera, FC2::Image  & grabbedImage );
int convertToBGRU( FC2::Image & image, FC2::Image & convertedImage );
int convertColorToMonoImage( FC2::Image & colorImage, FC2::Image & monoImage );
int generateTriclopsInput( FC2::Image const & grabbedImage, ImageContainer   & imageContainer, TriclopsInput& triclopsInput );
void StereoCycle(std::mutex* myImageMutex, StereoImage* mySharedImage);

void SendImageChunk();
void HandleCommand( const CrossPack*, NetTransMethod );
void StartMapScan(const CrossPack*, NetTransMethod );
void EndMapScan( const CrossPack*, NetTransMethod );
void ReceiveMap( const CrossPack*, NetTransMethod );
void Receive2D(const CrossPack*, NetTransMethod );
void ResetToHome( const CrossPack*, NetTransMethod );
void SendLoginInfo( const CrossPack*, NetTransMethod );
void MotorCycle();
void SensorCycle();
void PathingCycle();

void StopRobot();

void PlaySound(ESoundToPlay Sound, bool ShouldLoop = false);

void MoveToPoint( ArPose inPose );
std::list<ArPose> SimplifyPath(std::list<ArPose> path);

//Global vars
CrossClientID id;
unsigned short currentChunk = 0;
unsigned char currentFrame = 0;
bool hasInitializedImage = false;
bool shouldSendImage = false;
ArTime motorTimer;
ArTime lrfTimer;
ArTime bumperTimer;
ArTime sonarTimer;
ArTime imgChunkTimer;
ArTime positionTimer;
ArTime facingTimer;
short leftMotorVel;
short rightMotorVel;
ArRobot* myRobot;
ArDPPTU* myPTU;
CrossClient* myClient;
std::string filename;
ArLaserLogger* MyLogger;
ArLaser* myLaser;
string fileContent = "";
int amountOfFilePackets = 0;
ArSonarDevice* mySonar;
ArPathPlanningTask* myPathTask;
ArLocalizationTask* myLocTask;
ArPose currentGoal;
std::list<ArPose> currentPath;
ArActionGoto* myGotoPoseAction;
bool frontBumperState = false;
bool rearBumperState = false;
vector<uchar> compressedLeftFrame;
vector<uchar> compressedRightFrame;
StereoImage* sharedImage;
std::mutex* imageMutex;
vector<RobotSonar> sonars;
vector<RobotCamera> cameras;
RobotCamera currentLeftCamera;
RobotCamera currentRightCamera;
ArSoundsQueue* mySound = NULL;
ESoundToPlay LastSound = ESoundToPlay::None;
bool SoundIsLooping = false;

// config file vars
string serverIP; 		//i p address of the server
string friendlyName; 	// the display name for the bot
uchar robotTypeID; 		// the model type of bot
int limitStopDistance; 	// at what distance to stop the robot
int limitSlowDistance; 	// at what distance to use slowSpeed
int limitSlowSpeed; 	// set max speed when within slowDistance
float limitAvoidWidthRatio; // turning ratio durign auto object avoid
int normalCloseDistance; 	// base distance for if we are close to something
int normalSpeed; 		// base robot speed
int normalTurnSpeed; 	// normal speed for turning during pathing
float normalTurnAmount; // turn amount for a basic pathing turn
int lrfSegment; 		// step size for laser range finder divide
float lrfHeadingOffset;
float lrfOriginX;
float lrfOriginY;
float lrfOriginZ;
float minSpeed;
float maxSpeed;
int soundLevel = 1;

//Macro for handling triclops api errors
#define _HANDLE_TRICLOPS_ERROR( description, error )	\
{ \
   if( error != TriclopsErrorOk ) \
   { \
      printf( \
	 "*** Triclops Error '%s' at line %d :\n\t%s\n", \
	 triclopsErrorToString( error ), \
	 __LINE__, \
	 description );	\
      printf( "Press any key to exit...\n" ); \
      getchar(); \
      exit( 1 ); \
   } \
}

//**********************
//*************Arnl*****
//**********************

void logOptions(const char *progname)
{
  ArLog::log(ArLog::Normal, "Usage: %s [options]\n", progname);
  ArLog::log(ArLog::Normal, "[options] are any program options listed below, or any ARNL configuration");
  ArLog::log(ArLog::Normal, "parameters as -name <value>, see params/arnl.p for list.");
  ArLog::log(ArLog::Normal, "For example, -map <map file>.");
  Aria::logOptions();
}

bool gyroErrored = false;
const char* getGyroStatusString(ArRobot* robot)
{
  if(!robot || !robot->getOrigRobotConfig() || robot->getOrigRobotConfig()->getGyroType() < 2) return "N/A";
  if(robot->getFaultFlags() & ArUtil::BIT4)
  {
    gyroErrored = true;
    return "ERROR/OFF";
  }
  if(gyroErrored)
  {
    return "OK but error before";
  }
  return "OK";
}

// This function is called whenever a new goal is reached. It will be attached
// to the path planning task below in main() via ArPathPlanningTask::addGoalDoneCB()
void goalDone(ArPose goalPos) //, ArPathPlanningTask *pathTask)
{
  ArLog::log(ArLog::Normal, "ARNL server example: goal reached");
}

// This function is called if pathplanning fails. It will be attached
// to the path planning task below in main() via
// ArPathPlanningTask::addGoalFailedCB()
void goalFailed(ArPose goalPos) //, ArPathPlanningTask *pathTask)
{
  ArLog::log(ArLog::Normal, "ARNL server example: goal failed");
}

// This function is called if localization fails. It will be attached
// to the localization task below in main() via
// ArLocalizationTask::setFailedCallback()
void locFailed(int n) //, ArLocalizationTask* locTask)
{
  ArLog::log(ArLog::Normal, "ARNL server example: localization failed");
}

// This function is called whenever the path planning task changes its state
// (for example, from idle to planning a path, to following a planned path). It will be attached
// to the path planning task below in main() via
// ArPathPlanningTask::addStateChangeCB()
//ArPathPlanningInterface *pathPlanningTask = NULL;
void pathPlanStateChanged(ArPathPlanningTask *pathPlanningTask)
{
  char s[256];
  pathPlanningTask->getFailureString(s, 256);
  ArLog::log(ArLog::Normal, "ARNL server example: Path planning state: %s", s);
}

/// Log messages from robot controller
bool handleDebugMessage(ArRobotPacket *pkt)
{
  if(pkt->getID() != ArCommands::MARCDEBUG) return false;
  char msg[256];
  pkt->bufToStr(msg, sizeof(msg));
  msg[255] = 0;
  ArLog::log(ArLog::Terse, "Controller Firmware: %s", msg);
  return true;
}

//**********************
//***Custom Handlers****
//**********************

/* On ready event */
void HandleReady()
{
	printf("Ready to transmit!\n");
	id = myClient->GetClientID();
}

/* On disconnect event */
void HandleDisconnect()
{
	printf("Failed to connect/reconnect. Exiting..\n");
}

/* On trying to reconnect event */
void HandleAttemptReconnect()
{
	printf("Disconnected from server. Attempting to reconnect..\n");
	PlaySound(ESoundToPlay::Reconnection);
}

/* On failed reconnect event */
void HandleFailedReconnect()
{
	printf("Failed to reconnect! Re-initializing..\n");
}

/* On successful reconnect event */
void HandleSuccesfulReconnect()
{
	printf("Reconnected to server! Re-initializing..\n");
}

/* On connect to server event */
void HandleConnect()
{
	printf("Connected to server with ID: %d! Initializing..\n", myClient->GetClientID());
	id = myClient->GetClientID();
	PlaySound(ESoundToPlay::Connection);
}

/* On receive initial handshake request event */
void HandleHandshake()
{
	if (myClient->GetClientState() == CrossClientState::CLIENT_REQUESTING_ID)
		printf("Requesting old ID..\n");
	else
		printf("Requesting new ID..\n");
}

void HandleTransmitError(const CrossPack* pack, NetTransMethod method, NetTransError error)
{
	printf("Transfer error received via %s\n", (method == NetTransMethod::TCP ? "TCP" : "UDP"));

}

// Sound feedback handlers
void queueNowEmpty()
{
	if(SoundIsLooping && myClient && myClient->IsRunning()) {
		PlaySound(LastSound, true);
	}
}

void queueNowNonempty()
{
	// nada
}

//**********************
//*************Main*****
//**********************
int main(int argc, char **argv)
{
// Initialize Aria and Arnl global information
  Aria::init();
  Arnl::init();

  /* Mandatory cross sock initialization */
  CrossSockUtil::Init();

  /* Set client properties */
  CrossClientProperties props;
  props.maxConnectionAttempts = 10;
  props.maxReconnectionAttempts = 999;
  props.alivenessTestDelay = 2000.0;

  // our base client object
  CrossClient client = CrossClient(props);

  // set up our parser
  ArArgumentParser parser(&argc, argv);

  // load the default arguments
  parser.loadDefaultArguments();

  // The robot object
  ArRobot robot;

  // handle messages from robot controller firmware and log the contents
  robot.addPacketHandler(new ArGlobalRetFunctor1<bool, ArRobotPacket*>(&handleDebugMessage));

  // This object is used to connect to the robot, which can be configured via
  // command line arguments.
  ArRobotConnector robotConnector(&parser, &robot);

  // Connect to the robot
  if (!robotConnector.connectRobot())
  {
    ArLog::log(ArLog::Normal, "Error: Could not connect to robot... exiting");
    Aria::exit(3);
  }

  // Add a section to the configuration to change ArLog parameters
  ArLog::addToConfig(Aria::getConfig());

  // set up a gyro (if the robot is older and its firmware does not
  // automatically incorporate gyro corrections, then this object will do it)
  ArAnalogGyro gyro(&robot);

  // the laser connector
  ArLaserConnector laserConnector(&parser, &robot, &robotConnector);

  // used to connect to camera PTZ control
  ArPTZConnector ptzConnector(&parser, &robot);

  // Load default arguments for this computer (from /etc/Aria.args, environment
  // variables, and other places)
  parser.loadDefaultArguments();

  // Parse arguments
  if (!Aria::parseArgs() || !parser.checkHelpAndWarnUnparsed())
  {
    logOptions(argv[0]);
    Aria::exit(1);
  }


  // This causes Aria::exit(9) to be called if the robot unexpectedly
  // disconnects
  ArGlobalFunctor1<int> shutdownFunctor(&Aria::exit, 9);
  robot.addDisconnectOnErrorCB(&shutdownFunctor);


  // Create an ArSonarDevice object (ArRangeDevice subclass) and
  // connect it to the robot.
  ArSonarDevice sonarDev;
  robot.addRangeDevice(&sonarDev);
  mySonar = &sonarDev;


  // This object will allow robot's movement parameters to be changed through
  // a Robot Configuration section in the ArConfig global configuration facility.
  ArRobotConfig robotConfig(&robot);

  // Include gyro configuration options in the robot configuration section.
  robotConfig.addAnalogGyro(&gyro);

  // Start the robot thread.
  robot.runAsync(true);

  // connect the laser(s) if it was requested, this adds them to the
  // robot too, and starts them running in their own threads
  ArLog::log(ArLog::Normal, "Connecting to laser(s) configured in parameters...");
  if (!laserConnector.connectLasers())
  {
    ArLog::log(ArLog::Normal, "Error: Could not connect to laser(s). Exiting.");
    Aria::exit(2);
  }
  ArLog::log(ArLog::Normal, "Done connecting to laser(s).");

  // find the laser we should use for localization and/or mapping,
  // which will be the first laser
  robot.lock();
  ArLaser *firstLaser = robot.findLaser(1);
  if (firstLaser == NULL || !firstLaser->isConnected())
  {
    ArLog::log(ArLog::Normal, "Did not have laser 1 or it is not connected, cannot start localization and/or mapping... exiting");
    Aria::exit(2);
  }
  myLaser = firstLaser;
  robot.unlock();


//*************************
//************Our Stuff****
//*************************


  // ***********************************************************
  // Add the magical actions that will take us to victory OwO **
  // ***********************************************************

  // Read Config File settings for use with setup
  readConfigFile();

  // Collision avoidance actions at higher priority
  // Near limiter
  ArActionLimiterForwards limiterAction("speed limiter near", limitStopDistance, limitSlowDistance, limitSlowSpeed, limitAvoidWidthRatio);

  // Far limiter (if needed)
  //ArActionLimiterForwards limiterFarAction("speed limiter far", 200, 600, 450, 1);

  // TableSensor (if robot has one)
  //ArActionLimiterTableSensor tableLimiterAction;

  // Add limiters
  robot.addAction(&limiterAction, 95);
  //robot.addAction(&tableLimiterAction, 100);
  //robot.addAction(&limiterFarAction, 90);

  // Goto action at lower priority
  ArActionGoto gotoPoseAction("goto", ArPose(0.0, 0.0, 0.0), normalCloseDistance, normalSpeed, normalTurnSpeed, normalTurnAmount);
  robot.addAction(&gotoPoseAction, 50);
  myGotoPoseAction = &gotoPoseAction;

  // Stop action at lower priority, so the robot stops if it has no goal
  ArActionStop stopAction("stop");
  robot.addAction(&stopAction, 40);


  ArDataLogger dataLogger(&robot, "dataLog.txt");
  dataLogger.addToConfig(Aria::getConfig());

  // parse the command line... fail and print the help if the parsing fails
  // or if the help was requested
  if (!Aria::parseArgs() || !parser.checkHelpAndWarnUnparsed())
  {
    Aria::logOptions();
    Aria::exit(1);
  }

  client.SetReadyHandler(&HandleReady);
  client.SetDisconnectHandler(&HandleDisconnect);
  client.SetAttemptReconnectHandler(&HandleAttemptReconnect);
  client.SetReconnectHandler(&HandleSuccesfulReconnect);
  client.SetConnectHandler(&HandleConnect);
  client.SetReconnectFailedHandler(&HandleFailedReconnect);
  client.SetHandshakeHandler(&HandleHandshake);
  client.SetTransmitErrorHandler(&HandleTransmitError);

  ArIRs irs;
  robot.addRangeDevice(&irs);

  ArBumpers bumpers;
  robot.addRangeDevice(&bumpers);

  // store a reference to the local robot
  myRobot = &robot;

  // store a reference to the client object
  myClient = &client;

  /* Create and set up map object */

 // Set up the map object, this will look for files in the examples
 // directory (unless the file name starts with a /, \, or .
 // You can take out the 'fileDir' argument to look in the program's current directory
 // instead.
 // When a configuration file is loaded into ArConfig later, if it specifies a
 // map file, then that file will
 //loaded as the map.
 ArMap map;
 map.readFile("level.map");
 // set it up to ignore empty file names (otherwise if a configuration omits
 // the map file, the whole configuration change will fail)
 map.setIgnoreEmptyFileName(true);
 // ignore the case, so that if someone is using MobileEyes or
 // MobilePlanner from Windows and changes the case on a map name,
 // it will still work.
 map.setIgnoreCase(true);

 cout << endl << endl << "File name: "<< map.getFileName() << endl << endl;

  /* Create localization and path planning threads */

  ArPathPlanningTask pathTask(&robot, firstLaser, &sonarDev, &map);
  myPathTask = &pathTask;

// The following are callback functions that pathTask will call
  // on certain events such as reaching a goal, failing to reach a goal, etc.
  // The functions are defined at the top of this file. You can add code there
  // to take some action such as control something on the robot, choose the next
  // goal, send output elsewhele, etc.

  ArGlobalFunctor1<ArPose> goalDoneCB(&goalDone); //, &pathTask);
  pathTask.addGoalDoneCB(&goalDoneCB);

  ArGlobalFunctor1<ArPose> goalFailedCB(&goalFailed); //, &pathTask);
  pathTask.addGoalFailedCB(&goalFailedCB);

  ArGlobalFunctor1<ArPathPlanningTask*> stateCB(&pathPlanStateChanged, &pathTask);
  pathTask.addStateChangeCB(&stateCB);


  ArLog::log(ArLog::Normal, "Creating laser localization task");
  // Laser Monte-Carlo Localization
  ArLocalizationTask locTask(&robot, firstLaser, &map);
  myLocTask = &locTask;

  // A callback function, which is called if localization fails
  ArGlobalFunctor1<int> locFailedCB(&locFailed);
  locTask.setFailedCallBack(&locFailedCB); //, &locTask);


  // Set some options  and callbacks on each laser that the laser connector
  // connected to.
  std::map<int, ArLaser *>::iterator laserIt;
  for (laserIt = robot.getLaserMap()->begin();
       laserIt != robot.getLaserMap()->end();
       laserIt++)
  {
    int laserNum = (*laserIt).first;
    ArLaser *laser = (*laserIt).second;

    // Skip lasers that aren't connected
    if(!laser->isConnected())
      continue;

    // add the disconnectOnError CB to shut things down if the laser
    // connection is lost
    laser->addDisconnectOnErrorCB(&shutdownFunctor);
    // set the number of cumulative readings the laser will take
    laser->setCumulativeBufferSize(200);
    // add the lasers to the path planning task
    pathTask.addRangeDevice(laser, ArPathPlanningTask::BOTH);
    // set the cumulative clean offset (so that they don't all fire at once)
    laser->setCumulativeCleanOffset(laserNum * 100);
    // reset the cumulative clean time (to make the new offset take effect)
    laser->resetLastCumulativeCleanTime();

    // Add the packet count to the Aria info strings (It will be included in
    // MobileEyes custom details so you can monitor whether the laser data is
    // being received correctly)
    std::string laserPacketCountName;
    laserPacketCountName = laser->getName();
    laserPacketCountName += " Packet Count";
    Aria::getInfoGroup()->addStringInt(
	    laserPacketCountName.c_str(), 10,
	    new ArRetFunctorC<int, ArLaser>(laser,
					 &ArLaser::getReadingCount));
  }

  // set the path planning so it uses the explicit collision range for how far its planning
  pathTask.setUseCollisionRangeForPlanningFlag(true);

  // Add additional range devices to the robot and path planning task (so it avoids obstacles detected by these devices)

  // Add IR range device to robot and path planning task (so it avoids obstacles
  // detected by this device)
  robot.lock();
  pathTask.addRangeDevice(&irs, ArPathPlanningTask::CURRENT);

  // Add bumpers range device to robot and path planning task (so it avoids obstacles
  // detected by this device)
  pathTask.addRangeDevice(&bumpers, ArPathPlanningTask::CURRENT);

  // Add range device which uses forbidden regions given in the map to give virtual
  // range device readings to ARNL.  (so it avoids obstacles
  // detected by this device)
  ArForbiddenRangeDevice forbidden(&map);
  pathTask.addRangeDevice(&forbidden, ArPathPlanningTask::CURRENT);

  robot.unlock();

  // Action to slow down robot when localization score drops but not lost.
  ArActionSlowDownWhenNotCertain actionSlowDown(&locTask);
  pathTask.getPathPlanActionGroup()->addAction(&actionSlowDown, 140);

  // Action to stop the robot when localization is "lost" (score too low)
  ArActionLost actionLostPath(&locTask, &pathTask);
  pathTask.getPathPlanActionGroup()->addAction(&actionLostPath, 150);

  // Arnl uses this object when it must replan its path because its
  // path is completely blocked.  It will use an older history of sensor
  // readings to replan this new path.  This should not be used with SONARNL
  // since sonar readings are not accurate enough and may prevent the robot
  // from planning through space that is actually clear.
  ArGlobalReplanningRangeDevice replanDev(&pathTask);

  // Do an initial localization of the robot. ARNL and SONARNL try all the home points
  // in the map, as well as the robot's current odometric position, as possible
  // places the robot is likely to be at startup.   If successful, it will
  // also save the position it found to be the best localized position as the
  // "Home" position, which can be obtained from the localization task (and is
  // used by the "Go to home" network request).
  // MOGS instead just initializes at the current GPS position.
  // (You will stil have to drive the robot so it can determine the robot's
  // heading, however. See GPS Mapping instructions.)
  locTask.localizeRobotAtHomeBlocking();

  // make a new PTU object
  myPTU = new ArDPPTU(myRobot);

#ifdef PTZ_SERIAL_PORT
  ArSerialConnection *mySerialConnection = new ArSerialConnection;
  ArLog::log(ArLog::Normal, "dpptuExample: connecting to DPPTU over computer serial port %s.", PTZ_SERIAL_PORT);
  if(mySerialConnection->open(PTZ_SERIAL_PORT) != 0)
  {
	ArLog::log(ArLog::Terse, "dpptuExample: Error: Could not open computer serial port %s for DPPTU!", PTZ_SERIAL_PORT);
    Aria::exit(5);
  }
  myPTU->setDeviceConnection(mySerialConnection);
#endif

   myPTU->init();
   myPTU->resetCalib();
   myPTU->awaitExec();
   myPTU->regStatPower();
   myPTU->regMotPower();

  // initialize timers
  motorTimer.setToNow();
  lrfTimer.setToNow();
  sonarTimer.setToNow();
  bumperTimer.setToNow();
  imgChunkTimer.setToNow();
  positionTimer.setToNow();
  facingTimer.setToNow();

  // add callback for command requests
  client.AddDataHandler("Command", &HandleCommand);

  // add callback for scan requests
  client.AddDataHandler("StartScan", &StartMapScan);

  // add callback for .2D file requests
  client.AddDataHandler("EndScan", &EndMapScan);

  // add callback for updating map file requests
  client.AddDataHandler("MapFile", &ReceiveMap);

  // add callback for sending back 2D file
  client.AddDataHandler("2DFile", &Receive2D);

  // add callback for reseting pose to the nearest home point
  client.AddDataHandler("ROBITSetHome", &ResetToHome);

  // add callback for login info requests
  client.AddDataHandler("RequestLoginInfo", &SendLoginInfo);

  // start stereo image cycle on separate thread
  imageMutex = new std::mutex;
  sharedImage = new StereoImage();
  std::thread stereoCycleThread(StereoCycle, imageMutex, sharedImage);

  // Create the sound queue.
  ArSoundsQueue soundQueue;

  // Set WAV file callbacks
  soundQueue.setPlayFileCallback(ArSoundPlayer::getPlayWavFileCallback());
  soundQueue.setInterruptFileCallback(ArSoundPlayer::getStopPlayingCallback());

  // Notifications when the queue goes empty or non-empty.
  soundQueue.addQueueEmptyCallback(new ArGlobalFunctor(&queueNowEmpty));
  soundQueue.addQueueNonemptyCallback(new ArGlobalFunctor(&queueNowNonempty));

  // Lower default logging mode
  soundQueue.setLogLevel(ArLog::LogLevel::Normal);

  // Run the sound queue in a new thread and save a global reference
  soundQueue.runAsync();
  mySound = &soundQueue;

  // now let it spin off in its own thread
  client.Connect(serverIP); //call client connect

  robot.lock();
  robot.enableMotors();
  robot.unlock();

  while(client.IsRunning() && robot.isRunning())
  {
	// Update the client, which automatically receives incoming data and reconnects to the server
	client.Update();

	//call cycle callbacks each tick tock tick
	if(client.IsReady()) {
		MotorCycle();
		SensorCycle();
		PathingCycle();
	}
  }

  // prompt user that we are exiting
  printf("Exiting...\n");

  // play disconnection sound
  PlaySound(ESoundToPlay::Disconnection);
  // give it a chance to start playing
  CrossSysUtil::SleepMS(5);
  // wait for it to finish and then delete sound threads
  while (soundQueue.isPlaying() && LastSound == ESoundToPlay::Disconnection) {
	  CrossSysUtil::SleepMS(25);
  }

  mySound = NULL;
  soundQueue.stop();
  soundQueue.clearQueue();
  soundQueue.threadFinished();

  // stop client and cleanup CrossSock
  client.Disconnect();
  CrossSockUtil::CleanUp();

  // stop stereo image cycle
  imageMutex->lock();
  sharedImage->shouldQuit = true;
  imageMutex->unlock();
  stereoCycleThread.join();
  delete imageMutex;
  delete sharedImage;

  // stop aria
  Aria::exit(0);
}

void readConfigFile()
{
	//read from config file
	ifstream configFile("ROBITConfig.txt");

	//line read in the file
	string line;

	//clear sonar vector
	sonars.clear();

	while(getline(configFile, line))
	{
		if (line.substr(0, 1).compare("#") == 0)
		{
			//just a comment line, nothing to do
		}
		else if (line.substr(0, 9).compare("ServerIP:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			serverIP = readIn;
		}
		else if (line.substr(0, 10).compare("RobotName:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			friendlyName = readIn;
		}
		else if (line.substr(0, 12).compare("RobotTypeID:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			robotTypeID = (uchar)atoi(readIn.c_str());
		}
		else if (line.substr(0, 18).compare("LimitStopDistance:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			limitStopDistance = atoi(readIn.c_str());
		}
		else if (line.substr(0, 18).compare("LimitSlowDistance:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			limitSlowDistance = atoi(readIn.c_str());
		}
		else if (line.substr(0, 15).compare("LimitSlowSpeed:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			limitSlowSpeed = atoi(readIn.c_str());
		}
		else if (line.substr(0, 21).compare("LimitAvoidWidthRatio:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			limitAvoidWidthRatio = atof(readIn.c_str());
		}
		else if (line.substr(0, 20).compare("NormalCloseDistance:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			normalCloseDistance = atoi(readIn.c_str());
		}
		else if (line.substr(0, 12).compare("NormalSpeed:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			normalSpeed = atoi(readIn.c_str());
		}
		else if (line.substr(0, 16).compare("NormalTurnSpeed:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			normalTurnSpeed = atoi(readIn.c_str());
		}
		else if (line.substr(0, 17).compare("NormalTurnAmount:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			normalTurnAmount = atof(readIn.c_str());
		}
		else if (line.substr(0, 4).compare("LRF:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			lrfSegment = atoi(readIn.c_str());
			getline(stream, readIn, ' ');
			lrfOriginX = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			lrfOriginY = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			lrfOriginZ = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			lrfHeadingOffset = atof(readIn.c_str());
		}
		else if (line.substr(0, 6).compare("Sonar:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			float x = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float y = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float z = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float pitch = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float yaw = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float roll = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float range = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			int sonarID = atoi(readIn.c_str());

			sonars.push_back(RobotSonar(x, y, z, pitch, yaw, roll, range, sonarID));
		}
		else if (line.substr(0, 6).compare("Speed:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			minSpeed = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			maxSpeed = atof(readIn.c_str());
		}
		else if (line.substr(0, 7).compare("Camera:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			float x = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float y = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float z = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float pitch = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float yaw = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			float roll = atof(readIn.c_str());
			getline(stream, readIn, ' ');
			int cameraIndex = atoi(readIn.c_str());

			//add to camera array
			if(cameraIndex >= cameras.size()) {
				cameras.resize(cameraIndex + 1);
			}
			cameras[cameraIndex] = RobotCamera(x, y, z, pitch, yaw, roll, cameraIndex);
		}
		else if (line.substr(0, 11).compare("SoundLevel:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			soundLevel = atoi(readIn.c_str());
		}
		else if (line.substr(0, 7).compare("Volume:") == 0)
		{
			string readIn;
			stringstream stream(line);

			//skip id
			getline(stream, readIn, ' ');

			//read data
			getline(stream, readIn, ' ');
			double volume = atof(readIn.c_str());
			ArSoundPlayer::setVolume(volume);
		}
	}
}

void StopRobot()
{
	//Stop the robot
	currentPath.clear();
	myRobot->lock();
	myGotoPoseAction->cancelGoal();
	myRobot->stop();
	myRobot->unlock();

	//Clear the path
	CrossPack outPack;
	outPack.SetDataID(myClient->GetDataIDFromName("MoveToPoint"));
	outPack.AddToPayload<CrossClientID>(id);
	outPack.AddToPayload<int>(0);
	myClient->SendToServer(&outPack);

	//Log that we stopped
	printf("Emergency Stop\n");

	//Play a sound
	PlaySound(ESoundToPlay::EmergencyStop);
}

void SendLoginInfo( const CrossPack*, NetTransMethod )
{
	CrossPackPtr pack = myClient->CreatePack("ROBITConnection");
	pack->AddToPayload<uchar>(robotTypeID);
	pack->AddStringToPayload(friendlyName);
	myClient->SendToServer(pack);
}

void HandleCommand( const CrossPack * packet, NetTransMethod method)
{
     // get IDs
     CrossClientID senderID = packet->RemoveFromPayload<CrossClientID>();
     CrossClientID receiverID = packet->RemoveFromPayload<CrossClientID>();

	// get the command
	uchar command = packet->RemoveFromPayload<uchar>();

	switch(command)
	{
	 case 0: // direct motor command
	 {
		// update motor data
		double throttle = (packet->RemoveFromPayload<double>() * (maxSpeed - minSpeed)) + minSpeed;
		double forwardVal = packet->RemoveFromPayload<double>();
		double rightVal = packet->RemoveFromPayload<double>();

		// ensure we are moving away from triggered bumpers
		if (forwardVal > 0.0 && frontBumperState) {
			forwardVal = 0.0;
			rightVal = 0.0;
		}
		else if(forwardVal < 0.0 && rearBumperState) {
			forwardVal = 0.0;
			rightVal = 0.0;
		}
		else if(frontBumperState && rearBumperState) {
			forwardVal = 0.0;
			rightVal = 0.0;
		}

		// translate movement vectors to wheel velocities
		if (forwardVal == 0.0 && rightVal == 0.0) {
			leftMotorVel = 0.0;
			rightMotorVel = 0.0;
		}
		else if (forwardVal > 0.0 && rightVal == 0.0) {
			leftMotorVel = throttle;
			rightMotorVel = throttle;
		}
		else if (forwardVal < 0.0 && rightVal == 0.0) {
			leftMotorVel = -throttle;
			rightMotorVel = -throttle;
		}
		else if (forwardVal > 0.0 && rightVal > 0.0) {
			leftMotorVel = throttle;
			rightMotorVel = throttle / 2.0;
		}
		else if (forwardVal > 0.0 && rightVal < 0.0) {
			leftMotorVel = throttle / 2.0;
			rightMotorVel = throttle;
		}
		else if (forwardVal < 0.0 && rightVal > 0.0) {
			leftMotorVel = -throttle / 2.0;
			rightMotorVel = -throttle;
		}
		else if (forwardVal < 0.0 && rightVal < 0.0) {
			leftMotorVel = -throttle;
			rightMotorVel = -throttle / 2.0;
		}
		else if (forwardVal == 0.0 && rightVal > 0.0) {
			leftMotorVel = throttle / 4.0;
			rightMotorVel = -throttle / 4.0;
		}
		else if (forwardVal == 0.0 && rightVal < 0.0) {
			leftMotorVel = -throttle / 4.0;
			rightMotorVel = throttle / 4.0;
		}
		else {
			// Default don't move
			leftMotorVel = 0.0;
			rightMotorVel = 0.0;
		}

		// update wheel velocity
		if(leftMotorVel != 0.0 || rightMotorVel != 0.0) {
			if(!currentPath.empty())
			{
				StopRobot();
			}
			myRobot->lock();
			myRobot->setVel2(leftMotorVel, rightMotorVel);
			myRobot->unlock();

			// update motor timer
			motorTimer.setToNow();
		}
		else if(currentPath.empty()) {
		  myRobot->lock();
		  myRobot->stop();
		  myRobot->unlock();
		}

		break;
	 }

	 case 1: // basic information request
	 {
		//nothing yet...
	    break;
	 }

     case 2:  //controlling the camera facing
     {
        float deltaTilt = packet->RemoveFromPayload<float>();
    	float deltaPan = packet->RemoveFromPayload<float>();
    	if(deltaTilt != 0.0f) {
    		myPTU->tiltRel(deltaTilt);
    	}
    	if(deltaPan != 0.0f) {
    		myPTU->panRel(deltaPan);
    	}
        break;
     }
     case 3:    //reset the camera facing
     {
        myPTU->pan(0.0);
        myPTU->tilt(0.0);
        break;
     }

    case 4:    //sets point to pathfind to
    {
      double x = packet->RemoveFromPayload<double>() * 10.0;
      double y = packet->RemoveFromPayload<double>() * -10.0;
      currentGoal = ArPose(x, y, 0.0);

      cout << endl << endl << "X: " << currentGoal.getX() << "   Y: " << currentGoal.getY() <<
            "    Heading: " << currentGoal.getTh() << endl << endl;

      MoveToPoint(currentGoal);
	  break;
    }

    case 5:    //stop pathing
    {
    	cout << "\nMERGENCY! DANGER! STOPING!\n";
      StopRobot();
      break;
    }

	 default: break;
	}
}

void MotorCycle()
{
	if(motorTimer.mSecSince() > WHEEL_TIMEOUT && currentPath.empty())
	{
		motorTimer.setToNow();

		// timeout wheel movement
		myRobot->lock();
		myRobot->stop();
		myRobot->unlock();
	}
}

void PathingCycle()
{
	if(!currentPath.empty() && myGotoPoseAction->haveAchievedGoal())
	{
		currentPath.pop_front();
		if(!currentPath.empty()){
			cout << "Moving to: (" << currentPath.begin()->getX() << ", " << currentPath.begin()->getY() << ", " << currentPath.begin()->getTh() << ")\n";
			myGotoPoseAction->setGoal(*currentPath.begin());
		}
		else { // else at goal

			//Play a sound
			PlaySound(ESoundToPlay::Goal);

			//Send a packet
			CrossPackPtr pack = myClient->CreatePack("AtGoal");
			pack->AddToPayload<CrossClientID>(myClient->GetClientID());
			myClient->SendToServer(pack);
		}
	}
}

void SensorCycle()
{
     // send bumper data
	if(bumperTimer.mSecSince() > BUMPER_BROADCAST_DELAY)
	{
         // reset timer
         bumperTimer.setToNow();

         CrossPack packet;
		 packet.SetDataID(myClient->GetDataIDFromName("Bumper"));
		 packet.AddToPayload<CrossClientID>(id);

         // get bumper data
         myRobot->lock();
         frontBumperState = myRobot->isFrontBumperTriggered();
         rearBumperState = myRobot->isRearBumperTriggered();
	     myRobot->unlock();

	     // stop if either bumper is triggered
	     if(frontBumperState || rearBumperState) {
	    	 if(!currentPath.empty()) {
	    		 StopRobot();
	    	 }
	     }

	     // finish packet
	     char bumperFlag = 0;
	     if(frontBumperState)
	    	 bumperFlag = CrossSysUtil::SetBit(bumperFlag, 0);
		 else
			 bumperFlag = CrossSysUtil::ClearBit(bumperFlag, 0);
	     if(rearBumperState)
			 bumperFlag = CrossSysUtil::SetBit(bumperFlag, 1);
		 else
			 bumperFlag = CrossSysUtil::ClearBit(bumperFlag, 1);
	     packet.AddToPayload<char>(bumperFlag);

	     // close and send the packet
	     myClient->StreamToServer(&packet);
	}

     // send laser range finder data
	if(lrfTimer.mSecSince() > LRF_BROADCAST_DELAY)
	{
          // reset timer
          lrfTimer.setToNow();

          // lock the robot
          myRobot->lock();

          // assemble LRF data into packet
          CrossPack packet;
          packet.SetDataID(myClient->GetDataIDFromName("DistanceReadings"));
          packet.AddToPayload<CrossClientID>(id);
          packet.AddToPayload<short>(0); // sensor ID

          std::map<int, ArLaser*> *lasers = myRobot->getLaserMap();

          for(std::map<int, ArLaser*>::const_iterator i = lasers->begin(); i != lasers->end(); ++i)
          {
             //int laserIndex = (*i).first; //buy why only dream of sheep?
             ArLaser* laser = (*i).second;
             packet.AddToPayload<short>((180/LRF_SEGMENT));

          for(int j = 90; j > -90; j-= LRF_SEGMENT)
             {
                double angle = 0;
                double dist = laser->currentReadingPolar(j - LRF_SEGMENT, j, &angle);

                if(dist >= laser->getMaxRange())
	               dist = -1;

                // add origin to packet
                packet.AddToPayload<float>(lrfOriginX);
                packet.AddToPayload<float>(lrfOriginY);
                packet.AddToPayload<float>(lrfOriginZ);
                packet.AddToPayload<float>(0.0f); // pitch

                // add readings to packet
                packet.AddToPayload<float>(-angle + lrfHeadingOffset); // yaw
                packet.AddToPayload<float>(0.0f); // roll
                packet.AddToPayload<float>(dist / 10.0f);
             }
          }

          // unlock the robot
          myRobot->unlock();

	     // close and send the packet
          myClient->StreamToServer(&packet);
	}

     // send sonar data
	if(sonarTimer.mSecSince() > SONAR_BROADCAST_DELAY)
	{
          // reset timer
          sonarTimer.setToNow();

          // assemble the sonar data into a packet
          CrossPack packet;
          packet.SetDataID(myClient->GetDataIDFromName("DistanceReadings"));
          packet.AddToPayload<CrossClientID>(id);
          packet.AddToPayload<short>(1); // sensor ID

          // lock the robot
          myRobot->lock();

          // check for valid sensor readings
          int num = myRobot->getNumSonar();
          short numValid = 0;
          for(int i = 0; i < sonars.size(); i++) {
        	  if(sonars[i].sonarID < num) {
        		  sonars[i].lastReading = myRobot->getSonarRange(sonars[i].sonarID) / 10.0f;
        	      sonars[i].isValid = sonars[i].lastReading > 0.0f && sonars[i].lastReading < sonars[i].maxRange;
        	      if(sonars[i].isValid) {
        	    	  numValid++;
        	      }
        	  }
        	  else {
        		  sonars[i].isValid = false;
        	  }
          }

          // unlock the robot
          myRobot->unlock();

	     packet.AddToPayload<short>(numValid);
	     for(int i = 0; i < sonars.size(); i++)
	     {
	    	 if(sonars[i].isValid) {
				 packet.AddToPayload<float>(sonars[i].originX);
				 packet.AddToPayload<float>(sonars[i].originY);
				 packet.AddToPayload<float>(sonars[i].originZ);
				 packet.AddToPayload<float>(sonars[i].originPitch);
				 packet.AddToPayload<float>(sonars[i].originYaw);
				 packet.AddToPayload<float>(sonars[i].originRoll);
				 packet.AddToPayload<float>(sonars[i].lastReading);
	    	 }
	     }

	     // close and send the packet
	     myClient->StreamToServer(&packet);
	}

     // send img chunk data
	if(imgChunkTimer.mSecSince() > IMAGE_CHUNK_BROADCAST_DELAY)
	{
          // reset timer
          imgChunkTimer.setToNow();

          // send left and right image chunks
          SendImageChunk();
	}

     if(positionTimer.mSecSince() > POSITION_BROADCAST_DELAY)
     {
          positionTimer.setToNow();

          CrossPack packet;
          packet.SetDataID(myClient->GetDataIDFromName("Position"));

          packet.AddToPayload<CrossClientID>(id);

          myRobot->lock();
          packet.AddToPayload<float>(myRobot->getPose().getX() * 0.1f);
          packet.AddToPayload<float>(myRobot->getPose().getY() * -0.1f);
          packet.AddToPayload<float>(0.0f);
          packet.AddToPayload<float>(0.0f);
          packet.AddToPayload<float>(-myRobot->getPose().getTh());
          packet.AddToPayload<float>(0.0f);

          myRobot->unlock();

          //send to server UDP
          myClient->StreamToServer(&packet);
     }

     if(facingTimer.mSecSince() > FACING_BROADCAST_DELAY)
	 {
		  facingTimer.setToNow();

		  CrossPack packet;
		  packet.SetDataID(myClient->GetDataIDFromName("Facing"));

		  packet.AddToPayload<CrossClientID>(id);
		  packet.AddToPayload<float>(myPTU->getTilt());
		  packet.AddToPayload<float>(-myPTU->getPan());
		  packet.AddToPayload<float>(0.0f);

		  //send to server UDP
		  myClient->StreamToServer(&packet);
	 }
}

void SendImageChunk()
{
	CrossPack packet;
	packet.SetDataID(myClient->GetDataIDFromName("ImageChunk"));

	// Calculate bytes sent so far
	unsigned int bytesSoFar = currentChunk * CHUNK_SIZE;

    // If the previous image has finished...
    if ((bytesSoFar >= compressedRightFrame.size() && bytesSoFar >= compressedLeftFrame.size())
		 || !hasInitializedImage)
    {
    	// lower the flag..
    	shouldSendImage = false; // @@@ try get lock, check if new image,
    		//copy to compressed frames, and THEN raise this / unlock stuff

    	// try to lock stereo image data
    	if(imageMutex->try_lock())
    	{
    		// if we have a new image..
    		if(sharedImage->hasNewImage)
    		{
    			// lower new image flag
    			sharedImage->hasNewImage = false;

    			// copy compressed image buffers
    			compressedLeftFrame.clear();
    			compressedRightFrame.clear();
				for(int x = 0; x < sharedImage->leftBuf.size(); x++) {
					compressedLeftFrame.push_back(sharedImage->leftBuf[x]);
				}
				for(int x = 0; x < sharedImage->rightBuf.size(); x++) {
					compressedRightFrame.push_back(sharedImage->rightBuf[x]);
				}

				// Increment to the next Frame
				currentFrame++;

				// Reset the chunk counter
				currentChunk = 0;

				// Initialize camera positions to hardpoints
				if(cameras.size() > 0) {
					currentLeftCamera = cameras[0];
				}
				if(cameras.size() > 1) {
					currentRightCamera = cameras[1];
				}

				// Add the current robot position
				myRobot->lock();
				float robotX = myRobot->getPose().getX() / 10.0;
				float robotY = myRobot->getPose().getY() / -10.0;
				float robotH = -myRobot->getPose().getTh();
				myRobot->unlock();
				currentLeftCamera.originX += robotX;
				currentLeftCamera.originY += robotY;
				currentLeftCamera.originYaw += robotH;
				currentRightCamera.originX += robotX;
				currentRightCamera.originY += robotY;
				currentRightCamera.originYaw += robotH;

				// Add the ptu rotation
				currentLeftCamera.originPitch += myPTU->getTilt();
				currentLeftCamera.originYaw += -myPTU->getPan();
				currentRightCamera.originPitch += myPTU->getTilt();
				currentRightCamera.originYaw += -myPTU->getPan();

				// Raise flags
				hasInitializedImage = true;
				shouldSendImage = true;
    		}

    		// unlock stereo image data
    		imageMutex->unlock();
    	}
    }

    // if we still have image data to send..
    if(shouldSendImage)
    {
		// Calculate data size
		int dataSize = std::max<unsigned int>(0, std::min<unsigned int>(CHUNK_SIZE, compressedLeftFrame.size() - currentChunk * CHUNK_SIZE));
		 //std::cout<<"Data size for leftBuf is: " << dataSize << std::endl;

		// If there is data to send
		if(dataSize > 0)
		{
			// Fill the packet
			packet.AddToPayload<CrossClientID>(id);
			packet.AddToPayload<float>(currentLeftCamera.originX);
			packet.AddToPayload<float>(currentLeftCamera.originY);
			packet.AddToPayload<float>(currentLeftCamera.originZ);
			packet.AddToPayload<float>(currentLeftCamera.originPitch);
			packet.AddToPayload<float>(currentLeftCamera.originYaw);
			packet.AddToPayload<float>(currentLeftCamera.originRoll);
			packet.AddToPayload<unsigned char>(0);
			packet.AddToPayload<unsigned char>(currentFrame);
			packet.AddToPayload<unsigned short>(currentChunk);
			packet.AddToPayload<unsigned short>(CHUNK_SIZE);
			//@@ if resoltuion is increased increase # of bytestotal buff size
			packet.AddToPayload<unsigned short>(compressedLeftFrame.size());
			//exact bytes needed for this packet
			packet.AddToPayload<unsigned short>(dataSize);
			//data
			packet.AddDataToPayload((char *)&compressedLeftFrame[currentChunk * CHUNK_SIZE], dataSize);

			// Close and send the packet
			myClient->StreamToServer(&packet);
		}

		// Clear to packet
		packet.ClearPayload();

		// Calculate data size
		dataSize = std::max<unsigned int>(0, std::min<unsigned int>(CHUNK_SIZE, compressedRightFrame.size() - currentChunk * CHUNK_SIZE));
		 //std::cout<<"Data size for rightBuf is: " << dataSize << std::endl;

		// If there is data to send
		if(dataSize > 0)
		{
			// Fill the packet
			packet.AddToPayload<CrossClientID>(id);
			packet.AddToPayload<float>(currentRightCamera.originX);
			packet.AddToPayload<float>(currentRightCamera.originY);
			packet.AddToPayload<float>(currentRightCamera.originZ);
			packet.AddToPayload<float>(currentRightCamera.originPitch);
			packet.AddToPayload<float>(currentRightCamera.originYaw);
			packet.AddToPayload<float>(currentRightCamera.originRoll);
			packet.AddToPayload<unsigned char>(1);
			packet.AddToPayload<unsigned char>(currentFrame);
			packet.AddToPayload<unsigned short>(currentChunk);
			packet.AddToPayload<unsigned short>(CHUNK_SIZE);
			//@@ if resoltuion is increased increase # of bytestotal buff size
			packet.AddToPayload<unsigned short>(compressedRightFrame.size());
			//exact bytes needed for this packet
			packet.AddToPayload<unsigned short>(dataSize);
			//data
			packet.AddDataToPayload((char *)&compressedRightFrame[currentChunk * CHUNK_SIZE], dataSize);

			// Close and send the packet
			myClient->StreamToServer(&packet);
		}

		// Increment to the next row
		currentChunk++;
    }
}

void StereoCycle(std::mutex* myImageMutex, StereoImage* mySharedImage)
{
	// define local memory
	Mat rectifiedFrame;
	Mat disparityFrame;
	FC2::Camera camera;
	TriclopsContext triclops;
	FC2::Image grabbedImage;
	Mat leftFrame, rightFrame;
	FC2::Image leftImage, rightImage;
	FC2::Image rightColorImage, leftColorImage;
	bool shouldQuit = false;
	vector<uchar> leftBuf, rightBuf;

	// do stereo initialization
	initializeStereo(camera, triclops);

	// start pulling images
	while (!shouldQuit) {

		grabImage( camera, grabbedImage );
				FC2T::ErrorType fc2TriclopsError = FC2T::unpackUnprocessedRawOrMono16Image(
											   grabbedImage,
											   true, //assume little endian
											   rightImage,
											   leftImage );
		if (fc2TriclopsError != FC2T::ERRORTYPE_OK)
		{
			return;
		}

		//convert to color
		if ( convertToBGRU(rightImage, rightColorImage) )
		{
			return;
		}
		if ( convertToBGRU(leftImage, leftColorImage) )
		{
			return;
		}

		Size size( RESIZE_WIDTH, RESIZE_HEIGHT );

		leftFrame = Mat(leftColorImage.GetRows(), leftColorImage.GetCols(), CV_8UC3);
		rightFrame = Mat(rightColorImage.GetRows(), rightColorImage.GetCols(), CV_8UC3);

		for(unsigned int i = 0; i < leftColorImage.GetRows() * leftColorImage.GetCols(); i++)
		{
			leftFrame.data[i*3+0] = leftColorImage.GetData()[i*4+0];
			leftFrame.data[i*3+1] = leftColorImage.GetData()[i*4+1];
			leftFrame.data[i*3+2] = leftColorImage.GetData()[i*4+2];
			rightFrame.data[i*3+0] = rightColorImage.GetData()[i*4+0];
			rightFrame.data[i*3+1] = rightColorImage.GetData()[i*4+1];
			rightFrame.data[i*3+2] = rightColorImage.GetData()[i*4+2];
		}

		resize(leftFrame, leftFrame, size);
		resize(rightFrame, rightFrame, size);

		vector<int> param;

		param.push_back(cv::IMWRITE_JPEG_QUALITY);
		param.push_back(IMAGE_QUALITY);	//0-100 scale 0 being worst and 100 being best

		// update shared image with compressed stereo frames
		leftBuf.clear();
		rightBuf.clear();
		imencode(".jpg", leftFrame, leftBuf, param);
		imencode(".jpg", rightFrame, rightBuf, param);

		// lock the shared data
		myImageMutex->lock();

		// get if we should quit
		shouldQuit = mySharedImage->shouldQuit;

		// dont go any further if we should quit..
		if(!shouldQuit)
		{
			// copy to the shared image buffers
			mySharedImage->leftBuf.clear();
			mySharedImage->rightBuf.clear();
			for(int x = 0; x < leftBuf.size(); x++) {
				mySharedImage->leftBuf.push_back(leftBuf[x]);
			}
			for(int x = 0; x < rightBuf.size(); x++) {
				mySharedImage->rightBuf.push_back(rightBuf[x]);
			}

			// raise new image flag
			mySharedImage->hasNewImage = true;
		}

		// unlock shared data
		myImageMutex->unlock();
	}

	// disconnect stereo
	disconnectStereo(camera, triclops);
}

int initializeStereo(FC2::Camera & camera,
        			 TriclopsContext & triclops)
{
    // connect camera
    camera.Connect();

    // configure camera
    if ( configureCamera( camera ) )
    {
		return EXIT_FAILURE;
    }

    // generate the Triclops context
    if ( generateTriclopsContext( camera, triclops ) )
    {
		return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int configureCamera( FC2::Camera & camera )
{
    FC2T::ErrorType fc2TriclopsError;
    FC2T::StereoCameraMode mode = FC2T::TWO_CAMERA;
    fc2TriclopsError = FC2T::setStereoMode( camera, mode );
    if ( fc2TriclopsError )
    {
        return FC2T::handleFc2TriclopsError(fc2TriclopsError, "setStereoMode");
    }

    FC2::Error fc2Error = camera.StartCapture();
    if (fc2Error != FC2::PGRERROR_OK)
    {
	return FC2T::handleFc2Error(fc2Error);
    }

    return 0;
}

int generateTriclopsContext( FC2::Camera & camera,
                         TriclopsContext & triclops )
{
    FC2::CameraInfo camInfo;
    FC2::Error fc2Error = camera.GetCameraInfo(&camInfo);
    if (fc2Error != FC2::PGRERROR_OK)
    {
        return FC2T::handleFc2Error(fc2Error);
    }

    FC2T::ErrorType fc2TriclopsError;
    fc2TriclopsError = FC2T::getContextFromCamera( camInfo.serialNumber,
	                                                &triclops );
    if (fc2TriclopsError != FC2T::ERRORTYPE_OK)
    {
        return FC2T::handleFc2TriclopsError(fc2TriclopsError,
		                                    "getContextFromCamera");
    }

    return 0;
}

int grabImage ( FC2::Camera & camera, FC2::Image & grabbedImage )
{
	FC2::Error fc2Error = camera.RetrieveBuffer(&grabbedImage);
	if (fc2Error != FC2::PGRERROR_OK)
	{
		return FC2T::handleFc2Error(fc2Error);
	}

	return 0;
}

int convertToBGRU( FC2::Image & image, FC2::Image & convertedImage )
{
    FC2::Error fc2Error;
    fc2Error = image.SetColorProcessing(FC2::HQ_LINEAR);
    if (fc2Error != FC2::PGRERROR_OK)
    {
        return FC2T::handleFc2Error(fc2Error);
    }

    fc2Error = image.Convert(FC2::PIXEL_FORMAT_BGRU, &convertedImage);
    if (fc2Error != FC2::PGRERROR_OK)
    {
        return FC2T::handleFc2Error(fc2Error);
    }

    return 0;
}

int convertColorToMonoImage( FC2::Image & colorImage, FC2::Image & monoImage )
{
    FC2::Error fc2Error;
    fc2Error = colorImage.SetColorProcessing(FC2::HQ_LINEAR);
    if (fc2Error != FC2::PGRERROR_OK)
    {
        return FC2T::handleFc2Error(fc2Error);
    }

    fc2Error = colorImage.Convert(FC2::PIXEL_FORMAT_MONO8,
                                                 &monoImage);
    if (fc2Error != FC2::PGRERROR_OK)
    {
        return FC2T::handleFc2Error(fc2Error);
    }

    return 0;
}

// generate triclops input
int generateTriclopsInput( FC2::Image const & grabbedImage,
                           ImageContainer   & imageCont,
                           TriclopsInput    & triclopsInput )
{

    FC2::Error      fc2Error;
    FC2T::ErrorType fc2TriclopsError;
    TriclopsError   te;

    FC2::Image * tmpImage = imageCont.tmp;
    FC2::Image * unprocessedImage = imageCont.unprocessed;

    // Convert the pixel interleaved raw data to de-interleaved and color processed data
    fc2TriclopsError = FC2T::unpackUnprocessedRawOrMono16Image(
                                   grabbedImage,
								   true /*assume little endian*/,
                                   tmpImage[RIGHT],
                                   tmpImage[LEFT] );

    if (fc2TriclopsError != FC2T::ERRORTYPE_OK)
    {
	    return FC2T::handleFc2TriclopsError(fc2TriclopsError,
		                                   "unprocessedRawOrMono16Image()");
    }

    // check if the unprocessed image is color
    if ( tmpImage[0].GetBayerTileFormat() != FC2::NONE )
    {
   	    for ( int i = 0; i < 2; ++i )
   	    {
   	  	    if ( convertColorToMonoImage(tmpImage[i], unprocessedImage[i]) )
   	  	    {
   	  	  	    return 1;
   	  	    }
   	    }
    }
    else
    {
        unprocessedImage[RIGHT] = tmpImage[RIGHT];
        unprocessedImage[LEFT]  = tmpImage[LEFT];
    }

    // pack image data into a TriclopsInput structure
    te = triclopsBuildRGBTriclopsInput(
               grabbedImage.GetCols(),
               grabbedImage.GetRows(),
               grabbedImage.GetCols(),
               (unsigned long)grabbedImage.GetTimeStamp().seconds,
               (unsigned long)grabbedImage.GetTimeStamp().microSeconds,
               unprocessedImage[RIGHT].GetData(),
               unprocessedImage[LEFT].GetData(),
               unprocessedImage[LEFT].GetData(),
               &triclopsInput);

    _HANDLE_TRICLOPS_ERROR( "triclopsBuildRGBTriclopsInput()", te );

    return 0;
}

int disconnectStereo(FC2::Camera & camera,
        			 TriclopsContext & triclops)
{
    // Close the camera
    camera.StopCapture();
    camera.Disconnect();

    // Destroy the Triclops context
    TriclopsError     te;
    te = triclopsDestroyContext( triclops ) ;
    _HANDLE_TRICLOPS_ERROR( "triclopsDestroyContext()", te);

    return EXIT_SUCCESS;
}

void StartMapScan( const CrossPack * packet, NetTransMethod method)
{
  filename = "1scans.2d";

     // @@@ This starts a bunch of warnings because a packet handler doesn't consume the packet, and falls through to this client.
  MyLogger = new ArLaserLogger(myRobot, myLaser, 300, 25, filename.c_str(), true);
}

void EndMapScan( const CrossPack * packet, NetTransMethod method)
{
  //Stop scanning
  delete MyLogger;
  MyLogger = NULL;

  //Send the scan file (.2D) to the server
  CrossPackPtr outPack = myClient->CreatePack("2DFile");

  ifstream file(filename.c_str());
  myLocTask->localizeRobotAtHomeBlocking();

	string content = "";
	string line = "";

	while (getline(file, line))
	{
		content += line + "\n";
	}
	file.close();

	short numPackets = ceil(content.size() / (double)MAX_PACKET_SIZE);
	outPack->AddToPayload<short>(numPackets);
	myClient->SendToServer(outPack);

	outPack->ClearPayload();
	cout << numPackets << endl;

	for (int i = 0; i < (int)numPackets; i++)
	{
		string subStr = content.substr(i * MAX_PACKET_SIZE, MAX_PACKET_SIZE);
		outPack->AddStringToPayload(subStr);
		myClient->SendToServer(outPack);
		outPack->ClearPayload();
	}
}

void ReceiveMap( const CrossPack * packet, NetTransMethod method)
{
  if (amountOfFilePackets <= 0)
	{
	  amountOfFilePackets = packet->RemoveFromPayload<short>();
	  fileContent = "";
	}
	else
	{
		fileContent += packet->RemoveStringFromPayload();
		amountOfFilePackets--;

		if (amountOfFilePackets <= 0)
		{
			cout << "\nMapFile Received!" << endl;
			ofstream outputFile("level.map");
			outputFile << fileContent;
			outputFile.close();

			// update the map file
			if(myPathTask) {
				ArMapInterface* myMap =	myPathTask->getAriaMap();
				if(myMap) {
					myMap->readFile("level.map");
					myPathTask->setMapChangedFlag(true);
				}
			}
		}
	}
}

void Receive2D(const CrossPack * packet, NetTransMethod method)
{
  cout << "This should never, ever happen! :[\n";
}

void ResetToHome( const CrossPack * packet, NetTransMethod method)
{
  myLocTask->localizeRobotAtHomeBlocking();

  //play a sound
  PlaySound(ESoundToPlay::SendHome);
}


void MoveToPoint( ArPose inPose )
{
  //Reset the packet
  CrossPack outPack;
  outPack.SetDataID(myClient->GetDataIDFromName("MoveToPoint"));

  //Add ID to packet
  outPack.AddToPayload<CrossClientID>(id);

  //Get the path to the pose
  myRobot->lock();
  ArPose robotPose(myRobot->getPose().getX(), myRobot->getPose().getY(), myRobot->getPose().getTh());
  myRobot->unlock();
  std::list<ArPose> path = myPathTask->getPathFromTo(robotPose, currentGoal);

  //Apply linear regression to the path
  path = SimplifyPath(path);

  //Set the path to this current path
  currentPath = path;

  //Ensure the motors are enabled
  myRobot->lock();
  myRobot->enableMotors();
  myRobot->clearDirectMotion();
  myRobot->unlock();

  //Start moving to first point
  if(!currentPath.empty())
  {
	cout << "Moving to: (" << currentPath.begin()->getX() << ", " << currentPath.begin()->getY() << ", " << currentPath.begin()->getTh() << ")\n";
  	myGotoPoseAction->setGoal(*currentPath.begin());
  }

  //play a sound
  PlaySound(ESoundToPlay::PathingStart);

  //add path size to packet
  outPack.AddToPayload<int>(path.size());

  //add path data to packet
  std::list<ArPose>::const_iterator iterator;
  for (iterator = path.begin(); iterator != path.end(); ++iterator)
  {
     double x = iterator->getX() / 10.0;
     double y = iterator->getY() / -10.0;

	 outPack.AddToPayload<double>(x);
	 outPack.AddToPayload<double>(y);
	 outPack.AddToPayload<double>(0.0);
  }

  //Send the packet
  myClient->SendToServer(&outPack);
}

std::list<ArPose> SimplifyPath(std::list<ArPose> path)
{
  cout << "\n\nSize: " << path.size() << endl;

  if(path.size() < 2)
    return path;

  ArPose startPose = path.front();
  path.pop_front();
  ArPose previousPose;
  std::list<ArPose> simplifiedPath;
  std::list<ArPose> finalPath;
  simplifiedPath.push_back(startPose);
  double currentM = 0.0f;
  double lastM = 0.0f;
  bool hasLine = false;

  for (std::list<ArPose>::const_iterator iterator = path.begin(); iterator != path.end(); ++iterator)
  {
    currentM = (iterator->getY() - startPose.getY()) / (iterator->getX() - startPose.getX());

    if(currentM != lastM && hasLine)
      {
        simplifiedPath.push_back(previousPose);
        startPose = *iterator;
      }

    lastM = currentM;
    previousPose = *iterator;
    hasLine = true;
  }
  simplifiedPath.push_back(path.back());

  previousPose = simplifiedPath.front();
  finalPath.push_back(previousPose);
  for (std::list<ArPose>::const_iterator iterator = simplifiedPath.begin(); iterator != simplifiedPath.end(); ++iterator)
  {
    if(distanceFormula(previousPose.getX(), iterator->getX(), previousPose.getY(), iterator->getY()) >= 200.0)
    {
      finalPath.push_back(*iterator);
      previousPose = *iterator;
    }
  }


  cout << "\nOriginal Path: " << path.size() << "\tSimple Path: " << simplifiedPath.size() << "\tSimpler Path: " << finalPath.size() << "\n\n";
  return finalPath;
}

void PlaySound(ESoundToPlay Sound, bool ShouldLoop)
{
	if(soundLevel > 0) {
		LastSound = Sound;
		SoundIsLooping = ShouldLoop;

		if(mySound) {
			// Clear the queue of anything extra
			if(mySound->isPlaying()) {
				mySound->clearQueue();
			}

			// Figure out which sound to play for this event
			if(Sound == ESoundToPlay::None) {
				// nothing
			}
			else if(Sound == ESoundToPlay::Connection) {
				if(soundLevel == 1){
					mySound->play("Resources/Connection_1.wav");
				} else if(soundLevel == 2){
					mySound->play("Resources/Connection_2.wav");
				} else {
					mySound->play("Resources/Connection_3.wav");
				}
			}
			else if(Sound == ESoundToPlay::Disconnection) {
				if(soundLevel == 1){
					mySound->play("Resources/Disconnection_1.wav");
				} else if(soundLevel == 2){
					mySound->play("Resources/Disconnection_2.wav");
				} else {
					mySound->play("Resources/Disconnection_3.wav");
				}
			}
			else if(Sound == ESoundToPlay::Reconnection) {
				if(soundLevel == 1){
					mySound->play("Resources/Reconnection_1.wav");
				} else if (soundLevel == 2){
					mySound->play("Resources/Reconnection_2.wav");
				} else {
					mySound->play("Resources/Reconnection_3.wav");
				}
			}
			else if(Sound == ESoundToPlay::EmergencyStop) {
				if(soundLevel == 1){
					mySound->play("Resources/EmergencyStop_1.wav");
				} else if(soundLevel == 2){
					mySound->play("Resources/EmergencyStop_2.wav");
				} else {
					mySound->play("Resources/EmergencyStop_3.wav");
				}
			}
			else if(Sound == ESoundToPlay::PathingStart) {
				if(soundLevel == 2){
					mySound->play("Resources/PathingStart_2.wav");
				} else if(soundLevel >= 3){
					mySound->play("Resources/PathingStart_3.wav");
				}
			}
			else if(Sound == ESoundToPlay::Goal) {
				if(soundLevel == 2){
					mySound->play("Resources/Goal_2.wav");
				} else if(soundLevel >= 3){
					mySound->play("Resources/Goal_3.wav");
				}
			}
			else if(Sound == ESoundToPlay::SendHome) {
				if(soundLevel == 2){
					mySound->play("Resources/SendHome_2.wav");
				} else if(soundLevel >= 3){
					mySound->play("Resources/SendHome_3.wav");
				}
			}
		}
	}
}
