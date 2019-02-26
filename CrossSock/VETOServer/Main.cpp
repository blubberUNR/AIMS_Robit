#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\video.hpp>
#include <opencv2\videoio.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <math.h>
#include <sys/timeb.h>
#include <time.h>
#include <fstream>
#include <thread>
#include <mutex>
#include <conio.h>
#include <ctime>
#include <cmath>
#include "CrossServer.h"

using namespace cv;
using namespace std;
using namespace CrossSock;

#define MAX_PACKET_SIZE 1024
#define PROXIMITY_WARNING_THRESHOLD 500
#define IMAGE_BUFFER_SIZE 100000
#define CHUNK_SIZE 1024
#define IMAGE_PROC_FILTER_AMOUNT 4
#define ENCODE_QUALITY 80

// Identifies what data should be sent from a given ROBIT to all User clients
enum ESubscriptionLevel
{
	UNSUBSCRIBBED = 0,	// No data will be sent
	MINIMAL = 1,		// Only position data and connection notifications will be sent
	ALL_DATA = 2		// Position + sensor data will be sent
};

struct Position
{
public:

	double x;
	double y;
	double yaw;
	 
	Position()
	{
		x = 0;
		y = 0;
		yaw = 0;
	}

	Position(double inX, double inY, double inYaw)
	{
		x = inX;
		y = inY;
		yaw = inYaw;
	}

	Position(const Position& Other){
		x = Other.x;
		y = Other.y;
		yaw = Other.yaw;
	}

	virtual double Length()
	{
		return sqrt(x*x + y*y);
	}

	virtual void Normalize()
	{
		double length = Length();
		x /= length;
		y /= length;
	}

};

struct Position3D : public Position
{
public:

	double z;

	double pitch;
	double roll;

	Position3D()
	{
		x = 0;
		y = 0;
		z = 0;
		pitch = 0;
		yaw = 0;
		roll = 0;
	}

	Position3D(double inX, double inY, double inYaw)
	{
		x = inX;
		y = inY;
		yaw = inYaw;
	}

	Position3D(double inX, double inY, double inZ, double inPitch, double inYaw, double inRoll)
	{
		x = inX;
		y = inY;
		z = inZ;
		pitch = inPitch;
		yaw = inYaw;
		roll = inRoll;
	}

	Position3D(const Position3D& Other)
	{
		x = Other.x;
		y = Other.y;
		z = Other.z;
		pitch = Other.pitch;
		yaw = Other.yaw;
		roll = Other.roll;
	}

	double Length() override
	{
		return sqrt(x*x + y*y + z*z);
	}

	void Normalize() override
	{
		double length = Length();
		x /= length;
		y /= length;
		z /= length;
	}

};

struct DistanceReading
{
public:

	Position3D Origin;
	double Distance;

	DistanceReading()
	{
		Distance = 0.0;
	}

	DistanceReading(const Position3D& inOrigin, const double& inDistance)
	{
		Origin = inOrigin;
		Distance = inDistance;
	}
};

struct CameraData
{
public:

	Position3D Position;						// the position of the camera
	unsigned int frameIndex;					// what frame/image we are on
	unsigned int bytesReceived;					// how much of the total frame we have received
	unsigned char buffer[IMAGE_BUFFER_SIZE];	// buffer for the current frame
	Mat imageMat;								// holds a color image
	Mat maskMat;								// holds an image mask
	vector<uchar> compressedFrame;				// the compressed frame to be sent off
	Ptr<BackgroundSubtractor> pMOG2;			// background subtraction object from OpenCV
	bool bgSubInit;								// has the background subtraction object been set up for this camera?

	CameraData()
	{
		frameIndex = 0;
		bytesReceived = 0;
		bgSubInit = false;
	}
};


enum EClientType
{
	UNDEFINED_CLIENT,
	USER_CLIENT,
	ROBIT_CLIENT
};

enum ELogMessageType
{
	PROBLEM = 0,	// unexpected or bad behavior
	SERVER,			// event occuring or performed by the server
	TEXT,			// text/comment block for the log file
	INST,			// instruction sent/received
	PACKID,			// ID of a packet that was sent/received
	COMM,			// command packet
	LOC,			// location/position packet
	FACE			// "facing" pitch/yaw/roll
};

enum EImageProcessingMode
{
	NO_PROCESSING = 0,		// transmit only the color frames
	AUGMENTED_VIRTUALITY,	// transmit color and mask frames with BG subtraction method
	RUNTIME_NORMALS			// transmit only color and heighmap for runtime normal generation
};

/*
 * The global CrossServer reference - use this to access the server
 * this reference should be valid for the duration of the software
 */
CrossServer* myServer;

/*
* Global function prototype to delete a VETO client - defined above with VETOClient 
* forward decleration so that the VETOClient class has access to it!
*/
class VETOClient;
void RemoveVETOClient(VETOClient* vetoClient);


// parent client class for reflection
class VETOClient
{
protected:
	EClientType clientType;

	// private constructor on purpose so that this class cannot be constructed
	VETOClient()
	{
		clientType = EClientType::UNDEFINED_CLIENT;
	}

public:
	CrossClientID id;

	bool IsUserClient() const
	{
		return clientType == USER_CLIENT;
	}

	bool IsROBITClient() const
	{
		return clientType == ROBIT_CLIENT;
	}

	CrossClientEntryPtr GetClientEntry()
	{
		if (myServer) {
			CrossClientEntryPtr self = myServer->GetClientEntry(id);
			if (self) {
				return self;
			}
			
			// Something broke, remove/disconnect this client!
			RemoveVETOClient(this);
		}

		// Either the server or client was invalid, so just return null
		return nullptr;
	}
};

class UserClient : public VETOClient
{
public:
	unsigned char defaultSubscriptionLevel;
	vector<pair<CrossClientID, unsigned char>> explicitSubscriptionList;

	UserClient(CrossClientID idNumber)
	{
		clientType = EClientType::USER_CLIENT;
		id = idNumber;
		defaultSubscriptionLevel = ESubscriptionLevel::UNSUBSCRIBBED;
	}
};

class ROBITClient : public VETOClient
{
public:
	vector<UserClient*> minimallySubscribedUsers;		// contains the user clients that are at least minimally subscribed to this ROBIT
	vector<UserClient*> fullySubscribedUsers;			// contains the user clients that are fully subscribed to this ROBIT
	
	char bumperFlags;									// stores if the bumpers on the robot have been hit
	Position3D location;								// where the robot currently is
	Position3D facing;									// pitch/yaw/roll of a robot's "forward" vector
	vector<CameraData> cameras;							// robot camers
	vector<vector<DistanceReading>> distanceSensors;	// the SONAR/LRF sensors on the robot
	string name;										// the name we display for this robot
	CrossClientID typeID;
	vector<Position3D> path;
	string scanFileContent;
	int amountOfScanFilePackets;
	bool flush;											// if we need to flush the image buffers

	ROBITClient(CrossClientID idNumber, unsigned char type, string ROBITname)
	{
		clientType = EClientType::ROBIT_CLIENT;
		id = idNumber;
		name = ROBITname;
		typeID = type;
		scanFileContent = "";
		amountOfScanFilePackets = 0;
	}

	void AddClientToSubscriptionList(UserClient* Client, ESubscriptionLevel SubLevel) {
		if (Client) {
			if (SubLevel == ESubscriptionLevel::MINIMAL) {
				minimallySubscribedUsers.push_back(Client);
			}
			else if (SubLevel == ESubscriptionLevel::ALL_DATA) {
				fullySubscribedUsers.push_back(Client);
				minimallySubscribedUsers.push_back(Client);
			}
		}
	}
};

// Prototypes
void PackageImageChunk(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void HandleCommand(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void HandleDistanceReadings(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void HandleBumpers(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void CommandResult(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void SetSubscribedROBITs(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void GetAvailableROBITs(const CrossPack *, CrossClientEntryPtr, NetTransMethod);        //%%% suspect for reconnection bug
void UE4ClientConnection(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void ROBITConnection(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void SetROBITPosition(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void StartScan(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void EndScan(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void SendMapFile(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void Reading2DFile(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void SendArLevelFile(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void SendROBITsHome(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void MoveROBITToPoint(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void ROBITAtGoal(const CrossPack *, CrossClientEntryPtr, NetTransMethod);
void SetROBITFacing(const CrossPack *, CrossClientEntryPtr, NetTransMethod);

// Global variables
vector<ROBITClient*> ROBITClients;	// all connected robots
vector<UserClient*> UserClients;	// all connected UE4 Engine clients
string folderPath;					// stores the exe location for windows pathing use
string runningMode;					// what image mode the server is running in
//bool imageProcessingEnabled = true;		// if any image processing will be done on the server
//bool imageFilteringEnabled = true;		// if image processing is enabled, will the images be filtered

// text logger
ofstream logger;					// used to log extra data in time-stamped log file

// log threads
std::vector<std::thread*> logThreads;	// allows for log writing to be non-blocking

// logging flags
bool verbosePacket = false;			// enables logging of all incoming/outgoing packet IDs
bool verbosePosition = false;		// enables logging of all sent position data
bool verboseCommand = false;		// enables logging of all command packets and what was instructed
bool quitCheck = false;				// prevents accidental server shutdown

// extra server properties
EImageProcessingMode serverImageMode = NO_PROCESSING;

void updateScreen(string s = " ", bool bClear = true)
{
	// "clear" screen by shifting down
	for (int i = 0; i < 10; i++)
	{
		std::cout << "\n\n\n";
	}

	//check which mode we are in
	switch (serverImageMode)
	{
	case NO_PROCESSING:
		runningMode = "Normal";
		break;
	case AUGMENTED_VIRTUALITY:
		runningMode = "VR AR";
		break;
	case RUNTIME_NORMALS:
		runningMode = "Runtime Normals";
		break;
	}
		//runningMode = (imageProcessingEnabled == true) ? "ImageProc" : "Normal";

	// print menu and IDs
	std::cout << "\t\t   ******CrossServer v1.1******\n"
		<< "\t\t         FutureDev Branch!\n"
		<< "\t------------------------------------------------\n"
		<< "\t|P[" << verbosePacket << "]  V[" << verbosePosition << "]  C[" << verboseCommand << "]\n"
		<< "\t|Image Processing Mode: " << runningMode << "\n"
		<< "\t|Robot Clients:\n";
	for (auto &r : ROBITClients)
	{
		std::cout << "\t|\tRobot: Type[" << r->typeID << "] id[" << r->id << "] name[" << r->name << "]\n";
	}

	std::cout << "\t|\n\t|User Clients:\n";
	for (auto &u : UserClients)
	{
		std::cout << "\t|\tUE4 Client id[" << u->id << "]\n";
	}

	std::cout << "\t|\n\t------------------------------------------------\n";
		// print help commands
		std::cout << "\ts/e:\tstart/stop scanning\n\tm:\tmap file to robots\n\ta:\tarLevel to clients\n"
		<< "\th:\tsend all robits home\n\tp/v/c:\tpacket/position/command packet live display\n"
		<< "\ti:\ttoggle image processing\n\tf:\ttoggle image filtering\n"
		<< "\tk:\tshutdown server.\n\n";

		// print the passed string
		std::cout << s << "\n";
}

void logThread(ELogMessageType type, string s = "")
{
	string code;
	switch (type)
	{
	case PROBLEM:
		code = "ERROR";
		break;
	case SERVER:
		code = "SERVER";
		break;
	case TEXT:
		code = "TEXT";
		break;
	case INST:
		code = "INST";
		break;
	case PACKID:
		code = "PACKID";
		break;
	case COMM:
		code = "COMM";
		break;
	case LOC:
		code = "LOC";
		break;
	}

	char stime[9];
	_strtime_s(stime);

	logger << stime << "_" << code << ": " << s << "\n";
	return;
}

void logWrite(ELogMessageType type, string s = "")
{
	std::thread *t1 = new std::thread(logThread, type, s);

	logThreads.push_back(t1);
}

/* On server bind event */
void HandleBind()
{
	// nothing here
}

/* On client connected event */
void HandleNewClient(CrossClientEntryPtr client)
{
	// Nothing here. Was used for logging, moved to UE4/Robit connection handlers
}

/* When a client is ready to transmit */
void HandleClientReady(CrossClientEntryPtr client)
{
	VETOClient* vetoClient = client->GetCustomData<VETOClient>();

	// if we don't have valid client data, start/restart the login process
	if (!vetoClient) {
		CrossPackPtr pack = myServer->CreatePack("RequestLoginInfo");
		myServer->SendToClient(pack, client);
	}
	else { // otherwise, we are reconnecting
		
		if (vetoClient->IsROBITClient()) {
			ROBITClient* robitClient = (ROBITClient*)vetoClient;

			string s = "Reconnection attempt from Robit: " + to_string(client->GetClientID());

			// add back to connected ROBITs list
			ROBITClients.push_back(robitClient);

			// notify all user clients about this robit and initialize subscription level
			for (int j = 0; j < UserClients.size(); j++)
			{
				// update user clients with new available robit list
                //%%% make log entry here
				GetAvailableROBITs(nullptr, UserClients[j]->GetClientEntry(), NetTransMethod::TCP);

				// iterate through explicit list
				bool wasExplicit = false;
				for (int k = 0; k < UserClients[j]->explicitSubscriptionList.size(); k++) {

					if (robitClient->id == UserClients[j]->explicitSubscriptionList[k].first)
					{
						wasExplicit = true;
						robitClient->AddClientToSubscriptionList(UserClients[j], (ESubscriptionLevel)UserClients[j]->explicitSubscriptionList[k].second);
						break;
					}
				}

				// add to default if wasn't explicit
				if (!wasExplicit) {
					robitClient->AddClientToSubscriptionList(UserClients[j], (ESubscriptionLevel)UserClients[j]->defaultSubscriptionLevel);
				}
			}

			// log
			logWrite(SERVER, s);

			// update screen
			updateScreen(s);
		}
		else if (vetoClient->IsUserClient()) {
			UserClient* userClient = (UserClient*)vetoClient;

			string s = "Reconnection attempt from Client: " + to_string(client->GetClientID());

			// add back to connected user list
			UserClients.push_back(userClient);

			// iterate through all connected robits 
			for (int i = 0; i < ROBITClients.size(); i++)
			{
				// iterate through both sub lists and clear this user from both lists
				auto it = ROBITClients[i]->minimallySubscribedUsers.begin();
				while (it != ROBITClients[i]->minimallySubscribedUsers.end()) {
					if ((*it)->id == client->GetClientID()) {
						it = ROBITClients[i]->minimallySubscribedUsers.erase(it);
					}
					else {
						it++;
					}
				}
				it = ROBITClients[i]->fullySubscribedUsers.begin();
				while (it != ROBITClients[i]->fullySubscribedUsers.end()) {
					if ((*it)->id == client->GetClientID()) {
						it = ROBITClients[i]->fullySubscribedUsers.erase(it);
					}
					else {
						it++;
					}
				}

				// iterate through explicit list
				bool wasExplicit = false;
				for (int j = 0; j < userClient->explicitSubscriptionList.size(); j++) {

					if (ROBITClients[i]->id == userClient->explicitSubscriptionList[j].first)
					{
						wasExplicit = true;
						ROBITClients[i]->AddClientToSubscriptionList(userClient, (ESubscriptionLevel)userClient->explicitSubscriptionList[j].second);
						break;
					}
				}

				// add to default if wasn't explicit
				if (!wasExplicit) {
					ROBITClients[i]->AddClientToSubscriptionList(userClient, (ESubscriptionLevel)userClient->defaultSubscriptionLevel);
				}
			}

			// log
			logWrite(SERVER, s);

			// update screen
			updateScreen(s);
		}
	}
}

/* On client disconnected event */
void HandleDisconnect(CrossClientEntryPtr client)
{	
	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	RemoveVETOClient(vetoClient);
}

void HandleReconnect(CrossClientEntryPtr client)
{
	// nada!
}

void HandleDestroyClient(CrossClientEntryPtr client)
{
	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	if (vetoClient) {
		if (vetoClient->IsROBITClient()) {
			ROBITClient* robitClient = (ROBITClient*)vetoClient;
			client->SetCustomData<void>(nullptr);
			delete robitClient;
		}
		else if (vetoClient->IsUserClient()) {
			UserClient* userClient = (UserClient*)vetoClient;
			client->SetCustomData<void>(nullptr);
			delete userClient;
		}
	}
}

void HandleTransmitError(const CrossPack* pack, CrossClientEntryPtr client, NetTransMethod method, NetTransError error)
{
	printf("\nTransfer error received via %s. Error: %d\n", (method == NetTransMethod::TCP ? "TCP" : "UDP"), error);
	if (pack)
	{
		printf("Packet data ID: %d, Payload size: %d\n", pack->GetDataID(), pack->GetPayloadSize());
		string s = "Transfer Error. ID: " + to_string(pack->GetDataID());       //%%% need to approve with packet name instead of ID
		logWrite(PROBLEM, s);
	}
	printf("\n");
}

void HandleIncomingPacket(const CrossPack* pack, CrossClientEntryPtr client, NetTransMethod method)
{
	if (verbosePacket)
	{
		logWrite(PACKID, to_string(pack->GetDataID()));
		printf("Packet Received! DataID: %d\n", pack->GetDataID());
	}
}

void PutDown()
{
	// stop it!
	if (myServer) {
		myServer->Stop();
	}

	/* Mandatory cross sock deinitialization */
	CrossSockUtil::CleanUp();

	// delete all VETOClient objects
	for (auto it = UserClients.begin(); it != UserClients.end(); ++it) {
		delete *it;
	}
	for (auto it = ROBITClients.begin(); it != ROBITClients.end(); ++it) {
		delete *it;
	}

	// close the log file
	logger.close();

	// kill all threads
	auto itr = logThreads.begin();
	while (itr != logThreads.end()) {
		std::thread* t = *itr;
		itr = logThreads.erase(itr);
		if (t)
		{
			t->join();
			delete t;
		}
	}
}

int main(int argc, char **argv)
{
	// Mandatory cross sock initialization
	CrossSockUtil::Init();

	// put down the server on exit
	std::atexit(PutDown);

	// set server properties
	CrossServerProperties props;
	props.alivenessTestDelay = 2000.0;

	/* Set server events */
	CrossServer server(props);
	myServer = &server;
	server.SetServerBindHandler(&HandleBind);
	server.SetClientConnectedHandler(&HandleNewClient);
	server.SetClientReadyHandler(&HandleClientReady);
	server.SetDestroyClientHandler(&HandleDestroyClient);
	server.SetClientDisconnectedHandler(&HandleDisconnect);
	server.SetTransmitErrorHandler(&HandleTransmitError);
	server.SetReceiveDataHandler(&HandleIncomingPacket);
	server.SetClientReconnectedHandler(&HandleReconnect);

	// functors
	server.AddDataHandler("ImageChunk", &PackageImageChunk);
	server.AddDataHandler("DistanceReadings", &HandleDistanceReadings);
	server.AddDataHandler("Bumper", &HandleBumpers);
	server.AddDataHandler("Command", &HandleCommand);
	server.AddDataHandler("CommandResult", &CommandResult);
	server.AddDataHandler("SetSubscribedROBITs", &SetSubscribedROBITs);
	server.AddDataHandler("GetAvailableROBITs", &GetAvailableROBITs);
	server.AddDataHandler("UE4ClientConnection", &UE4ClientConnection);
	server.AddDataHandler("ROBITConnection", &ROBITConnection);
	server.AddDataHandler("Position", &SetROBITPosition);
	server.AddDataType("ROBITDisconnect");					// NOTE: Server needs to know about all data types, even those that it can't / will never receive!
	server.AddDataHandler("StartScan", &StartScan);
	server.AddDataHandler("EndScan", &EndScan);
	server.AddDataHandler("MapFile", &SendMapFile);
	server.AddDataHandler("2DFile", &Reading2DFile);
	server.AddDataHandler("ArLevel", &SendArLevelFile);
	server.AddDataHandler("ROBITSetHome", &SendROBITsHome);
	server.AddDataHandler("MoveToPoint", &MoveROBITToPoint);
	server.AddDataHandler("AtGoal", &ROBITAtGoal);
	server.AddDataType("ProximityWarning");					// NOTE: Server needs to know about all data types, even those that it can't / will never receive!
	server.AddDataType("RequestLoginInfo");					// NOTE: Server needs to know about all data types, even those that it can't / will never receive!
	server.AddDataHandler("Facing", &SetROBITFacing);

	server.Start(7425); // not sure if real or only dreams... tacos?

	// create text file for logging
	// date and time
	char sdate[9];
	char stime[9];
	_strdate_s(sdate);
	_strtime_s(stime);

	// start creating the log directory
	// get the path to this .exe
	char exePath[MAX_PATH];
	HMODULE hModule = GetModuleHandle(NULL);
	if (hModule != NULL)
	{
		GetModuleFileName(hModule, exePath, sizeof(exePath));
	}

	// convert to string and store in global value
	folderPath = exePath;

	// remove the exe from the path
	folderPath = folderPath.substr(0, folderPath.size()-15);

	// append log folder
	string logPath = folderPath + "\\Logs";

	// check if this folder already exists
	if ((GetFileAttributes(logPath.c_str())) == INVALID_FILE_ATTRIBUTES)
	{
		// directory doesn't exist, create it
		CreateDirectory(logPath.c_str(), 0);
	}

	// add log file filename
	string filename = logPath + "\\CrossServerLog_";
	filename.append(stime);
	filename.append("_");
	filename.append(sdate);
	filename.append(".txt");

	// remove unwanted characters
	for (int i = 2; i < filename.length(); ++i){
		if (filename[i] == '/' || filename[i] == ':')
			filename[i] = '-';
	}

	// create the log, add startup
	logger.open(filename);

	// write the help section of the log
	logWrite(TEXT, "CrossSock Server Log. Entires are listed by event type in all capitals.");
	logWrite(TEXT, "ERROR codes or bad server commands, TEXT for general logs, SERVER for server commands, INST for user/robot instructions, PACKID for CrossSock IDs for each sent packet, COMM for command packet info, LOC for robot location/heading packet info.\n");

	// server startup
	logWrite(SERVER, "Server Startup...");

	// print the main screen and help commands
	updateScreen();

	// input storage var
	char input = ' ';

	while (server.IsRunning()) // @@@ $$$ TODO: no scaling since we only care about one robot currently
	{
		/* Update the server, which automatically receives incoming data and connects to new clients */
		server.Update();

		// Get the current input, only if a key is pressed
		if (_kbhit())
		{
			input = (char)_getch();
		}

		if (input != ' ')
		{
			switch (input)
			{
			case 'p':
				verbosePacket = (verbosePacket == false) ? true : false;
				if (verbosePacket) updateScreen("Verbose Packet Log Enabled!");
				else updateScreen("Verbose Packet Log Disabled...", false);
				input = ' ';
				quitCheck = false;
				break;
			case 'v':
				verbosePosition = (verbosePosition == false) ? true : false;
				if (verbosePosition) updateScreen("Verbose Position Log Enabled!");
				else updateScreen("Verbose Position Log Disabled...", false);
				input = ' ';
				quitCheck = false;
				break;
			case 'c':
				verboseCommand = (verboseCommand == false) ? true : false;
				if (verboseCommand) updateScreen("Verbose Command Log Enabled!");
				else updateScreen("Verbose Command Log Disabled...", false);
				input = ' ';
				quitCheck = false;
				break;
			case 's':
				if (ROBITClients.size() > 0)
				{
					updateScreen("Starting Scan...");
					logWrite(SERVER, "Scan started from server");
					CrossPack broadcastPacket;
					broadcastPacket.SetDataID(myServer->GetDataIDFromName("StartScan"));
					for (int i = 0; i < ROBITClients.size(); i++)
					{
						myServer->SendToClient(&broadcastPacket, ROBITClients[i]->GetClientEntry());
					}
				}
				else {
					logWrite(PROBLEM, "Scan from server with no robot connected");
					updateScreen("ERROR: Scan cannot start without a connected robot.");
				}
				input = ' ';
				quitCheck = false;
				break;
			case 'e':
				if (ROBITClients.size() > 0)
				{
					updateScreen("Ending Scan...");
					logWrite(SERVER, "Server is ending scan");
					CrossPack broadcastPacket;
					broadcastPacket.SetDataID(myServer->GetDataIDFromName("EndScan"));
					for (int i = 0; i < ROBITClients.size(); i++)
					{
						myServer->SendToClient(&broadcastPacket, ROBITClients[i]->GetClientEntry());
					}
				}
				else {
					logWrite(PROBLEM, "Scan ended from server without connected robot");
					updateScreen("ERROR: Cannot end scan without a connected robot.");
				}
				input = ' ';
				quitCheck = false;
				break;
			case 'm':
				if (ROBITClients.size() > 0)
				{
					updateScreen("Sending Map File...");
					logWrite(SERVER, "Server sending map file to all robits");
					SendMapFile(new CrossPack, ROBITClients.front()->GetClientEntry(), NetTransMethod::TCP);
				}
				else {
					logWrite(PROBLEM, "Server sent map file to all robits with none connected");
					updateScreen("ERROR: Cannot send map file without a connected robot.");
				}
				input = ' ';
				quitCheck = false;
				break;
			case 'h':
				if (ROBITClients.size() > 0)
				{
					updateScreen("Syncing all ROBITs to nearest home points.");
					logWrite(SERVER, "Server is sending all robits home");
					CrossPack broadcastPacket;
					broadcastPacket.SetDataID(myServer->GetDataIDFromName("ROBITSetHome"));
					for (int i = 0; i < ROBITClients.size(); i++)
					{
						myServer->SendToClient(&broadcastPacket, ROBITClients[i]->GetClientEntry());
					}
				}
				else {
					logWrite(PROBLEM, "Server sent all robits home with none connected");
					updateScreen("ERROR: Cannot establish a \"Home\" point without a connected robot.");
				}
				input = ' ';
				quitCheck = false;
				break;
			case 'a':
				if (UserClients.size() > 0)
				{
					logWrite(SERVER, "Server sending ArLevel file to all UE4 clients");
					updateScreen("Sending ArLevel File...");
					SendArLevelFile(nullptr, UserClients.front()->GetClientEntry(), NetTransMethod::TCP);
				}
				else {
					logWrite(PROBLEM, "Sever sent ArLevel file with no UE4 clients connected");
					updateScreen("ERROR: Cannot send ARLevel without a connected UE4 client.");
				}
				input = ' ';
				quitCheck = false;
				break;
			case 'i':
				// cycle through image processing modes
				switch (serverImageMode)
				{
				case NO_PROCESSING:
					serverImageMode = AUGMENTED_VIRTUALITY;
					logWrite(SERVER, "Augmented Virtuality Processing Enabled");
					updateScreen("Augmented Virtuality processing Enabled.");
					break;
				case AUGMENTED_VIRTUALITY:
					serverImageMode = RUNTIME_NORMALS;
					logWrite(SERVER, "Runtime Normals Processing Enabled");
					updateScreen("Runtime Normals processing Enabled.");
					break;
				case RUNTIME_NORMALS:
					serverImageMode = NO_PROCESSING;
					logWrite(SERVER, "Image Processing Disabled");
					updateScreen("Image processing disabled.");
					break;
				}
				input = ' ';
				break;
			case 'k':
				// shutdown prompt
				if (!quitCheck)
				{
					updateScreen("Are you sure? Type 'k' again to shutdown.");
					quitCheck = true;
					input = ' ';
					break;
				}
				// they hit 'k' twice in a row, shutdown the server
				logWrite(SERVER, "Server Shutdown Started...");
				updateScreen("*****Server Shutdown Starting******");
				input = ' ';
				PutDown();
				break;
			}
		}
	} // end while(server.IsRunning())

	// cleanup everything else before shutdown
	PutDown();

	return 0;
}


void PackageImageChunk(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	// Grab the unique ID from whichever robot sent this packet
	CrossClientID robitID = packet->RemoveFromPayload<CrossClientID>();

	// Cast generic client pointer to a more specific VETO-pointer
	VETOClient* vetoClient = client->GetCustomData<VETOClient>();

	// Check if this pointer is valid and if it's a ROBIT client pointer
	if (vetoClient && vetoClient->IsROBITClient()) {

		// We know it's a robit, can cast it safely
		ROBITClient* robitClient = (ROBITClient*)vetoClient;

		// Make chunk counter
		int currentChunk = 0;

		// No image processing enabled, just forward the packet to the client ASAP
		if (serverImageMode == NO_PROCESSING) {
			for (int j = 0; j < robitClient->fullySubscribedUsers.size(); j++) {
				myServer->StreamToClient(packet, robitClient->fullySubscribedUsers[j]->GetClientEntry());
			}
		}

		// Get info from packet in prep for toy code
		float x = packet->RemoveFromPayload<float>();
		float y = packet->RemoveFromPayload<float>();
		float z = packet->RemoveFromPayload<float>();
		float pitch = packet->RemoveFromPayload<float>();
		float yaw = packet->RemoveFromPayload<float>();
		float roll = packet->RemoveFromPayload<float>();
		unsigned char camID = packet->RemoveFromPayload<unsigned char>();
		unsigned char frameNum = packet->RemoveFromPayload<unsigned char>();
		unsigned short chunkNum = packet->RemoveFromPayload<unsigned short>();
		unsigned short chunkSize = packet->RemoveFromPayload<unsigned short>();
		unsigned short bufSize = packet->RemoveFromPayload<unsigned short>();
		unsigned short dataSize = packet->RemoveFromPayload<unsigned short>();

		// Avoid packets with no data
		if (dataSize <= 0) {
			return;
		}

		// Adjust cameras array size on toy robot if too small
		if (camID >= robitClient->cameras.size()) {
			robitClient->cameras.resize(camID + 1);
		}

		// Reset data counter and sync frame number if on a new frame
		if (frameNum != robitClient->cameras[camID].frameIndex)
		{
			robitClient->cameras[camID].bytesReceived = 0;
			robitClient->cameras[camID].frameIndex = frameNum;
		}

		// Iif received data from the current frame, remove data from payload of dataSize, put into buffer at correct position
		if (frameNum == robitClient->cameras[camID].frameIndex)
		{
			packet->RemoveDataFromPayload((char*)&robitClient->cameras[camID].buffer[chunkSize * chunkNum], dataSize);
			robitClient->cameras[camID].Position = Position3D(x, y, z, pitch, yaw, roll);
			robitClient->cameras[camID].bytesReceived += dataSize;
		}

		// AR mode, we have received a full image, time to process
		if (serverImageMode == AUGMENTED_VIRTUALITY) 
		{
			// $$$
			// This will all be replaced with UHV BG Subtraction
			// 
			// Needs to send off a compressed color and mask frame to client

			// Full image received
			if (robitClient->cameras[camID].bytesReceived >= bufSize) {
				// move data from buffer into temp mat
				Mat tempMat(1, bufSize, CV_8UC1, robitClient->cameras[camID].buffer);

				// Decode mat and store in robitClient imageMat
				robitClient->cameras[camID].imageMat = imdecode(tempMat, CV_LOAD_IMAGE_COLOR);

				// Create streaming packet
				CrossPack fPack;
				fPack.SetDataID(myServer->GetDataIDFromName("ImageChunk"));

				// compression params, $$$ testing, not used at the moment
				/*
				int jpegquality = ENCODE_QUALITY;
				vector < int > compression_params;
				compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
				compression_params.push_back(jpegquality);
				*/

				// process the color and mask frame, one at a time
				for (int i = 0; i < 2; i++) {

					//chunk counter to use for this image
					currentChunk = 0;

					if (i == 0) {
						// Process color

						// Compress completed color frame
						if (imencode(".jpg", robitClient->cameras[camID].imageMat, robitClient->cameras[camID].compressedFrame)) {
							// success, write to file on server if desired for debug
							// imwrite("compressedColorFrame.jpg", robitClient->cameras[camID].compressedFrame);
						}
						else {
							// Failed, log failure
							ostringstream os;
							os << "Robit " << robitClient->id << " frame imencode failure!";
							string logS = os.str();
							updateScreen(logS);
							logWrite(PROBLEM, logS);
							return;
						}
					}
					else {
						// Process the mask

						// If BG subtraction needs to run setup
						if (!robitClient->cameras[camID].bgSubInit) {
							robitClient->cameras[camID].bgSubInit = true;
							robitClient->cameras[camID].pMOG2 = createBackgroundSubtractorMOG2(500, 16.0f, false);
						}

						// Create mask using MOG2
						robitClient->cameras[camID].pMOG2->apply(robitClient->cameras[camID].imageMat, robitClient->cameras[camID].maskMat);

                        // Run mophology filter on the mask
                        cv::Mat structElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(IMAGE_PROC_FILTER_AMOUNT, IMAGE_PROC_FILTER_AMOUNT));
                        cv::morphologyEx(robitClient->cameras[camID].maskMat, robitClient->cameras[camID].maskMat, cv::MORPH_OPEN, structElement);
                        cv::morphologyEx(robitClient->cameras[camID].maskMat, robitClient->cameras[camID].maskMat, cv::MORPH_CLOSE, structElement);

						// Compress mask frame
						if (imencode(".jpg", robitClient->cameras[camID].maskMat, robitClient->cameras[camID].compressedFrame)) {
							// success, write to file on server if desired for debug
							// imwrite("compressedMaskFrame.jpg", robitClient->cameras[camID].compressedFrame);
						}
						else {
							// Failed, log failure
							ostringstream os;
							os << "Robit " << robitClient->id << " mask frame imencode failure!";
							string logS = os.str();
							updateScreen(logS);
							logWrite(PROBLEM, logS);
							return;
						}
					}

					// We have finished processing either the color or mask frame, send it off
					
					// Pre-init dataSize to enter into loop
					int dataSize = max(0, min(CHUNK_SIZE, (int)robitClient->cameras[camID].compressedFrame.size() - (currentChunk * CHUNK_SIZE)));

					// Time to pack up the color image and stream it
					// While we still have data left to pack and send
					while (dataSize > 0) {
						// Re-calc remaining data size
						dataSize = max(0, min(CHUNK_SIZE, (int)robitClient->cameras[camID].compressedFrame.size() - (currentChunk * CHUNK_SIZE)));

						// If we still have data to send
						if (dataSize > 0) {

							// frame chunk and ID data
							fPack.AddToPayload<CrossClientID>(robitID);
							fPack.AddToPayload<float>(x);
							fPack.AddToPayload<float>(y);
							fPack.AddToPayload<float>(z);
							fPack.AddToPayload<float>(pitch);
							fPack.AddToPayload<float>(yaw);
							fPack.AddToPayload<float>(roll);
							fPack.AddToPayload<unsigned char>(camID);
							fPack.AddToPayload<unsigned char>(frameNum);
							fPack.AddToPayload<unsigned short>(currentChunk);
							fPack.AddToPayload<unsigned short>(CHUNK_SIZE);
							// if resoltuion is increased increase # of bytestotal buff size
							fPack.AddToPayload<unsigned short>(robitClient->cameras[camID].compressedFrame.size());
							// exact bytes needed for this packet
							fPack.AddToPayload<unsigned short>(dataSize);
							// data
							fPack.AddDataToPayload((char *)&robitClient->cameras[camID].compressedFrame[currentChunk * CHUNK_SIZE], dataSize);
							// if this is a mask, flag it!
							if (i > 0) {
								fPack.SetPacketFlag(CUSTOM_FLAG_1, true);
							}
							else {
								fPack.SetPacketFlag(CUSTOM_FLAG_1, false);
							}

							//Stream the packet
							for (int j = 0; j < robitClient->fullySubscribedUsers.size(); j++) {
								myServer->StreamToClient(&fPack, robitClient->fullySubscribedUsers[j]->GetClientEntry());
							}

							//next chunk
							currentChunk++;

							//clear payload so we can re-use packet
							fPack.ClearPayload();
						}
					}
				}
			}
		}

		// Runtime normal generation, we have a full image, time to process
		if (serverImageMode == RUNTIME_NORMALS)
		{
			if (robitClient->cameras[camID].bytesReceived >= bufSize)
			{
				// $$$ Add filter code here
				//
				// Needs to send off a compressed color and heightmap frame to client, both as COLOR images (R,G,B,A).
				// Sobel filter X and Y directions, combine on magnitiude, send off as heightmap. The Normalmap is made client-side.
			}
		}

		// $$$ @@@
        // TODO: Multi-threading needed, especially for BG subtraction. Supposed to be HPC, after all
		// Use UHV BG instead of OpenCV
        // don't use jpg compression? Or use better compression properties?
        // flush image buffers/etc on connection? Might not be needed though since we know the bug is elsewhere
	}
}

void HandleDistanceReadings(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	// pull of robit ID
	CrossClientID robitID = packet->RemoveFromPayload<CrossClientID>();

	// get robit
	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	if (vetoClient && vetoClient->IsROBITClient()) {
		ROBITClient* robitClient = (ROBITClient*)vetoClient;

		// forward packet to fully subscribbed clients
		for (int j = 0; j < robitClient->fullySubscribedUsers.size(); j++)
		{
			myServer->StreamToClient(packet, robitClient->fullySubscribedUsers[j]->GetClientEntry());
		}

		// get info about this sensor
		int sensorId = packet->RemoveFromPayload<short>();

		int amountOfReadings = packet->RemoveFromPayload<short>();
		
		// resize or clear sensor listing
		while (sensorId >= robitClient->distanceSensors.size()) {
			vector<DistanceReading> newSensor;
			robitClient->distanceSensors.push_back(newSensor);
		}

		// clear out old readings
		robitClient->distanceSensors[sensorId].clear();

		// for each reading
		for (int k = 0; k < amountOfReadings; k++)
		{
			// read distance reading
			double originX = packet->RemoveFromPayload<float>();
			double originY = packet->RemoveFromPayload<float>();
			double originZ = packet->RemoveFromPayload<float>();
			double originPitch = packet->RemoveFromPayload<float>();
			double originYaw = packet->RemoveFromPayload<float>();
			double originRoll = packet->RemoveFromPayload<float>();
			double distance = packet->RemoveFromPayload<float>();
			DistanceReading newReading(Position3D(originX, originY, originZ, originPitch, originYaw, originRoll), distance);

			// add distance reading
			robitClient->distanceSensors[sensorId].push_back(newReading);

			// send distance readings
			if (distance <= PROXIMITY_WARNING_THRESHOLD) {

				// assemble packet
				CrossPack broadcastPacket;
				broadcastPacket.SetDataID(myServer->GetDataIDFromName("ProximityWarning"));
				broadcastPacket.AddToPayload<CrossClientID>(robitID);
				broadcastPacket.AddToPayload<float>(originX);
				broadcastPacket.AddToPayload<float>(originY);
				broadcastPacket.AddToPayload<float>(originZ);
				broadcastPacket.AddToPayload<float>(originPitch);
				broadcastPacket.AddToPayload<float>(originYaw);
				broadcastPacket.AddToPayload<float>(originRoll);
				broadcastPacket.AddToPayload<float>(distance);

				// stream to minimally subscribbed clients
				for (int j = 0; j < robitClient->minimallySubscribedUsers.size(); j++)
				{
					myServer->StreamToClient(&broadcastPacket, robitClient->minimallySubscribedUsers[j]->GetClientEntry());
				}
			}
		}
	}
}

void HandleBumpers(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	CrossClientID robitID = packet->RemoveFromPayload<CrossClientID>();

	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	if (vetoClient && vetoClient->IsROBITClient()) {
		ROBITClient* robitClient = (ROBITClient*)vetoClient;

		for (int j = 0; j < robitClient->minimallySubscribedUsers.size(); j++)
		{
			myServer->StreamToClient(packet, robitClient->minimallySubscribedUsers[j]->GetClientEntry());
		}

		robitClient->bumperFlags = packet->RemoveFromPayload<char>();
	}
}

void HandleCommand(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	CrossClientID senderID = packet->RemoveFromPayload<CrossClientID>();
	CrossClientID recieverID = packet->RemoveFromPayload<CrossClientID>();
	CrossClientEntryPtr receiverClient = myServer->GetClientEntry(recieverID);

	if (receiverClient) {
		if (method == NetTransMethod::TCP) {
			myServer->SendToClient(packet, receiverClient);
		}
		else {
			myServer->StreamToClient(packet, receiverClient);
		}
	}

	// logging  (if enabled)
	if (verboseCommand)
	{
		ostringstream os;
		os << "Command was sent from " + senderID << " to " << recieverID << ".";
		string logS = os.str();

		// screen print as well (no updateScreen())
		cout << logS << endl;

		logWrite(COMM, logS);
	}
}

void SetROBITPosition(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	CrossClientID robitID = packet->RemoveFromPayload<CrossClientID>();

	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	if (vetoClient && vetoClient->IsROBITClient()) {
		ROBITClient* robitClient = (ROBITClient*)vetoClient;

		for (int j = 0; j < robitClient->minimallySubscribedUsers.size(); j++)
		{
			myServer->StreamToClient(packet, robitClient->minimallySubscribedUsers[j]->GetClientEntry());
		}

		double x = packet->RemoveFromPayload<float>();
		double y = packet->RemoveFromPayload<float>();
		double z = packet->RemoveFromPayload<float>();
		double pitch = packet->RemoveFromPayload<float>();
		double yaw = packet->RemoveFromPayload<float>();
		double roll = packet->RemoveFromPayload<float>();

		robitClient->location = Position3D(x, y, z, pitch, yaw, roll);

		// logging (if enabled)
		if (verbosePosition)
		{
			ostringstream os;
			os << "Robit " << robitID << " position: (x/y/z)(p/y/r) (" << x << ", " << y << ", " << z << ")  (" << pitch << ", " << yaw << ", " << roll << ").";
			string logS = os.str();

			// screen print as well (no updateScreen())
			cout << logS << endl;

			logWrite(LOC, logS);
		}
	}
}

void CommandResult(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	// Not sure if needed, but leave in and keep empty, jolly-o neat-o boy-o!
}

void SetSubscribedROBITs(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	CrossClientID idNumber = client->GetClientID();

	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	if (vetoClient && vetoClient->IsUserClient()) {
		UserClient* userClient = (UserClient*)vetoClient;

		// set default subscription level
		userClient->defaultSubscriptionLevel = packet->RemoveFromPayload<unsigned char>();

		// get all of the explicit subs from the list
		userClient->explicitSubscriptionList.clear();
		int amount = packet->RemoveFromPayload<short>();
		for (int i = 0; i < amount; i++)
		{
			CrossClientID robitID = packet->RemoveFromPayload<CrossClientID>();
			unsigned char subLevel = packet->RemoveFromPayload<unsigned char>();
			userClient->explicitSubscriptionList.push_back(pair<CrossClientID, unsigned char>(robitID, subLevel));
		}

		// iterate through all connected robits 
		for (int i = 0; i < ROBITClients.size(); i++)
		{
			// iterate through both sub lists and clear this user from both lists
			auto it = ROBITClients[i]->minimallySubscribedUsers.begin();
			while (it != ROBITClients[i]->minimallySubscribedUsers.end()) {
				if ((*it)->id == idNumber) {
					it = ROBITClients[i]->minimallySubscribedUsers.erase(it);
				}
				else {
					it++;
				}
			}
			it = ROBITClients[i]->fullySubscribedUsers.begin();
			while (it != ROBITClients[i]->fullySubscribedUsers.end()) {
				if ((*it)->id == idNumber) {
					it = ROBITClients[i]->fullySubscribedUsers.erase(it);
				}
				else {
					it++;
				}
			}

			// iterate through explicit list
			bool wasExplicit = false;
			for (int j = 0; j < userClient->explicitSubscriptionList.size(); j++) {

				if (ROBITClients[i]->id == userClient->explicitSubscriptionList[j].first)
				{
					wasExplicit = true;
					ROBITClients[i]->AddClientToSubscriptionList(userClient, (ESubscriptionLevel)userClient->explicitSubscriptionList[j].second);
					break;
				}
			}

			// add to default if wasn't explicit
			if (!wasExplicit) {
				ROBITClients[i]->AddClientToSubscriptionList(userClient, (ESubscriptionLevel)userClient->defaultSubscriptionLevel);
			}
			
		}

		logWrite(INST, "SetSubscribedRobits from id: " + to_string(userClient->id));
	}
}

void GetAvailableROBITs(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method) // reform the entire connected robit's list, called on a new robit conneciton or disconnection
{
	// Since this function can be called explicitly, we need to check if the client is valid!
	if (!client)
		return;

	CrossPack broadcastPacket;
	broadcastPacket.SetDataID(myServer->GetDataIDFromName("GetAvailableROBITs"));

	broadcastPacket.AddToPayload<short>(ROBITClients.size());

	for (int i = 0; i < ROBITClients.size(); i++)
	{
		broadcastPacket.AddToPayload<CrossClientID>(ROBITClients[i]->id);
		broadcastPacket.AddToPayload<CrossClientID>(ROBITClients[i]->typeID);
		broadcastPacket.AddStringToPayload(ROBITClients[i]->name.c_str());
	}

	myServer->SendToClient(&broadcastPacket, client);

	logWrite(INST, "GetAvailableRobits from id: " + to_string(client->GetClientID()));
}

void UE4ClientConnection(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	// assemble packet
	CrossPack broadcastPacket;
	broadcastPacket.SetDataID(myServer->GetDataIDFromName("UE4ClientConnection"));

	// store id
	CrossClientID id = client->GetClientID();

	// create user client and add to list
	UserClient* NewClient = new UserClient(id);
	UserClients.push_back(NewClient);

	// store user client as custom data
	client->SetCustomData<UserClient>(NewClient);

	// send id to client
	broadcastPacket.AddToPayload<CrossClientID>(id);
	myServer->SendToClient(&broadcastPacket, client);

	logWrite(SERVER, "UE4 Connection with id: " + to_string(id));

	updateScreen("New UE4 connection with id: " + to_string(id));
}

void ROBITConnection(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	// assemble packet
	CrossPack broadcastPacket;
	broadcastPacket.SetDataID(myServer->GetDataIDFromName("ROBITConnection"));

	// get robit data
	CrossClientID id = client->GetClientID();
	CrossClientID typeID = packet->RemoveFromPayload<uchar>();
	string name = packet->RemoveStringFromPayload();

	// create robit client and add to list
	ROBITClient* NewClient = new ROBITClient(id, typeID, name);
	ROBITClients.push_back(NewClient);

	// store robit client as custom data
	client->SetCustomData<ROBITClient>(NewClient);

	// send id to client
	broadcastPacket.AddToPayload<CrossClientID>(id);
	myServer->SendToClient(&broadcastPacket, client);

	// notify all user clients about this robit and initialize subscription level
	for (int j = 0; j < UserClients.size(); j++)
	{
		// update user clients with new available robit list
		GetAvailableROBITs(nullptr, UserClients[j]->GetClientEntry(), method);

		// iterate through explicit list
		bool wasExplicit = false;
		for (int k = 0; k < UserClients[j]->explicitSubscriptionList.size(); k++) {

			if (NewClient->id == UserClients[j]->explicitSubscriptionList[k].first)
			{
				wasExplicit = true;
				NewClient->AddClientToSubscriptionList(UserClients[j], (ESubscriptionLevel)UserClients[j]->explicitSubscriptionList[k].second);
				break;
			}
		}

		// add to default if wasn't explicit
		if (!wasExplicit) {
			NewClient->AddClientToSubscriptionList(UserClients[j], (ESubscriptionLevel)UserClients[j]->defaultSubscriptionLevel);
		}
	}

	logWrite(SERVER, "ROBIT Connection with id: " + to_string(id));

	updateScreen("New Robit connection with id: " + to_string(id));
}

void SendROBITsHome(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	CrossPack broadcastPacket;
	broadcastPacket.SetDataID(myServer->GetDataIDFromName("ROBITSetHome"));

	CrossClientID robitId = packet->RemoveFromPayload<CrossClientID>();
	CrossClientEntryPtr robitClient = myServer->GetClientEntry(robitId);
	if (robitClient) {
		myServer->SendToClient(&broadcastPacket, robitClient);
	}

	updateScreen("--------ROBIT Home Point Established--------");

	logWrite(INST, "Robit sent home, id: " + to_string(robitId));
}

void StartScan(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method) // TODO: scale issue
{
	CrossPack broadcastPacket;
	broadcastPacket.SetDataID(myServer->GetDataIDFromName("StartScan"));

	CrossClientID robitId = packet->RemoveFromPayload<CrossClientID>();
	CrossClientEntryPtr robitClient = myServer->GetClientEntry(robitId);
	if (robitClient) {
		myServer->SendToClient(&broadcastPacket, robitClient);
	}

	updateScreen("-------------Scan Started-------------");

	logWrite(INST, "Robit scan started, id: " + to_string(robitId));
}

void EndScan(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method) // TODO: scale issue
{
	CrossPack broadcastPacket;
	broadcastPacket.SetDataID(myServer->GetDataIDFromName("EndScan"));

	CrossClientID robitId = packet->RemoveFromPayload<CrossClientID>();
	CrossClientEntryPtr robitClient = myServer->GetClientEntry(robitId);
	if (robitClient) {
		myServer->SendToClient(&broadcastPacket, robitClient);
	}

	updateScreen("-------------Scan Ending-------------");

	logWrite(INST, "Robit scan ended, id: " + to_string(robitId));
}

void SendMapFile(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	// Since this function can be called explicitly, we need to check if the client is valid!
	if (!client)
		return;

	CrossPack broadcastPacket;
	broadcastPacket.SetDataID(myServer->GetDataIDFromName("MapFile"));

	string mapPath = folderPath + "\\mapFile.map";
	ifstream mapFile(mapPath);

	string content = "";
	string line = "";

	while (std::getline(mapFile, line))
	{
		content += line + "\n";
	}
	mapFile.close();

	short numPackets = (ceil(content.size() / (double)MAX_PACKET_SIZE));
	broadcastPacket.AddToPayload<short>(numPackets);
	for (int j = 0; j < ROBITClients.size(); j++)
	{
		myServer->SendToClient(&broadcastPacket, ROBITClients[j]->GetClientEntry());
	}
	broadcastPacket.ClearPayload();

	for (int i = 0; i < (int)numPackets; i++)
	{
		string subStr = content.substr(i * MAX_PACKET_SIZE, MAX_PACKET_SIZE);
		broadcastPacket.AddStringToPayload(subStr);

		for (int j = 0; j < ROBITClients.size(); j++)
		{
			myServer->SendToClient(&broadcastPacket, ROBITClients[j]->GetClientEntry());
		}

		broadcastPacket.ClearPayload();
	}

	updateScreen("-------------Map File Sent-------------");

	logWrite(INST, "Map file sent to all connected robits");
}

void Reading2DFile(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	if (vetoClient && vetoClient->IsROBITClient()) {
		ROBITClient* robitClient = (ROBITClient*)vetoClient;

		if (robitClient->amountOfScanFilePackets <= 0)
		{
			robitClient->amountOfScanFilePackets = packet->RemoveFromPayload<short>();
			robitClient->scanFileContent = "";
		}
		else
		{
			robitClient->scanFileContent += packet->RemoveStringFromPayload();
			robitClient->amountOfScanFilePackets--;

			if (robitClient->amountOfScanFilePackets <= 0)
			{
				std::string fileName = folderPath + "\\mapFile_";
				fileName += std::to_string(robitClient->id);
				fileName += ".2d";
				ofstream outputFile(fileName);
				outputFile << robitClient->scanFileContent;
				outputFile.close();
				string s = "Received 2D scan file.\nFilename: " + fileName;
				updateScreen(s);
				logWrite(SERVER, "2D Map file read");
			}
		}
	}
}

void SendArLevelFile(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	// Since this function can be called explicitly, we need to check if the client is valid!
	if (!client)
		return;

	CrossPack broadcastPacket;
	broadcastPacket.SetDataID(myServer->GetDataIDFromName("ArLevel"));

	string mapPath = folderPath + "\\mapFile.map";
	ifstream mapFile(mapPath);

	string content = "";
	string line = "";
	bool foundLINES = false;
	int vert = 0;
	int hor = 0;
	double wallPadding = 2;			// wall thickness modifier, in cm~uu
	double ceilingHeight = 284.5;	// how high the (highest) main ceiling should be (Z), in cm~uu
	double floorHeight = -5;		// height of the floor (Z), in cm~uu
	double floorThickness = 10;		// thickness of the ceiling mesh, in cm~uu
	double ceilingThickness = 10;	// thickness of floor mesh, in cm~uu
	int extraExpansion = 1500;		// how much further to push the walls and ceiling meshes past the level bounds;

	while (std::getline(mapFile, line))
	{
		if (line.compare("DATA") == 0)
			break;
		if (foundLINES)
		{
			string stuff;
			stringstream ss(line);

			std::getline(ss, stuff, ' ');
			int x1 = round(atoi(stuff.c_str()) / 10.0); // from mm to cm
			std::getline(ss, stuff, ' ');
			int y1 = round(atoi(stuff.c_str()) / 10.0); // keep normal for distance/math, flip sign later
			std::getline(ss, stuff, ' ');
			int x2 = round(atoi(stuff.c_str()) / 10.0);
			std::getline(ss, stuff);
			int y2 = round(atoi(stuff.c_str()) / 10.0); // keep normal for distance/math, flip sign later

			double z = ceilingHeight / 2; // half of height
			double deltaX = x2 - x1;
			double deltaY = y2 - y1;
			double angle = atan2(deltaY, deltaX);
			double midX = (deltaX / 2.0) + x1;
			double midY = (deltaY / 2.0) + y1;
			angle = angle * (180.0 / 3.14);		// convert back to degrees
			double length = sqrt(pow(deltaX, 2) + pow(deltaY, 2));
			midY = -midY;	// flip the sign of Y for UE4 coords
			angle = -angle; //	flip the sign of Yaw for UE4 coords

			// x y z l w h
			ostringstream os;
			os << "ww " << midX << " " << midY << " " << z << " " << length << " " << wallPadding << " " << angle << " " << ceilingHeight << "\n";
			content += os.str();
		}
		else
		{
			if (line.substr(0, 7).compare("MinPos:") == 0)
			{
				// this reads the MinPos and the next line, MaxPos. The full map corner points
				string stuff;
				stringstream ss(line);

				// skip id tag
				std::getline(ss, stuff, ' ');

				// get x,y of bottom left corner of map, MinPos
				std::getline(ss, stuff, ' ');
				int x1 = round(atoi(stuff.c_str()) / 10.0);
				std::getline(ss, stuff, ' ');
				int y1 = round(atoi(stuff.c_str()) / -10.0);

				// next point is on next line of file, get new line
				std::getline(mapFile, line);
				stringstream ss2(line);

				// skip id tag
				std::getline(ss2, stuff, ' ');

				// get x,y of top right corner of map, MaxPos
				std::getline(ss2, stuff, ' ');
				int x2 = round(atoi(stuff.c_str()) / 10.0); // from mm to cm
				std::getline(ss2, stuff, ' ');
				int y2 = round(atoi(stuff.c_str()) / -10.0);

				// simple math for length/width and center x/y coord
				int l = abs(x2 - x1);
				int w = abs(y2 - y1);
				int x = x1 + (l / 2);
				int y = y2 + (w / 2);
				l += extraExpansion;  // just to make sure it covers everything
				w += extraExpansion;

				// write to file, floor then ceiling
				ostringstream os;
				os << "fo " << x << " " << y << " " << floorHeight << " " << l << " " << w << " " << floorThickness << "\n";
				content += os.str();

				ostringstream os2;
				os2 << "ce " << x << " " << y << " " << ceilingHeight << " " << l << " " << w << " " << ceilingThickness << "\n";
				content += os2.str();
			}
			else if (line.substr(0, 20).compare("Cairn: ForbiddenArea") == 0)
			{
				string stuff;
				stringstream ss(line);

				// skip id content
				for (int i = 0; i < 8; i++) std::getline(ss, stuff, ' ');

				// WARNING!! Heading on forbidden areas must be 0.0 degrees in the map file!!
				// get the two point pairs
				std::getline(ss, stuff, ' ');
				int x1 = round(atoi(stuff.c_str()) / 10.0); // from mm to cm
				std::getline(ss, stuff, ' ');
				int y1 = round(atoi(stuff.c_str()) / 10.0);
				std::getline(ss, stuff, ' ');
				int x2 = round(atoi(stuff.c_str()) / 10.0);
				std::getline(ss, stuff);
				int y2 = round(atoi(stuff.c_str()) / 10.0);

				// simple math for length/width and origin x/y coord
				double deltaX = x2 - x1;
				double deltaY = y2 - y1;
				double midX = (deltaX / 2.0) + x1;
				double midY = (deltaY / 2.0) + y1;
				int z = (ceilingHeight/2) + 1; // half of height

				// flip y due to Mapper3 writing a negitive y to it's file... for reasons no one knows
				midY = -midY;

				// x and y center point and box dimensions
				ostringstream os;
				os << "fa " << midX << " " << midY << " " << z << " " << abs(deltaX) << " " << abs(deltaY) << " " << ceilingHeight << "\n";
				content += os.str();
			}
			else if (line.substr(0, 20).compare("Cairn: ForbiddenLine") == 0)
			{
				string stuff;
				stringstream ss(line);
				for (int i = 0; i < 8; i++) std::getline(ss, stuff, ' ');

				std::getline(ss, stuff, ' ');
				int x1 = round(atoi(stuff.c_str()) / 10.0); // from mm to cm
				std::getline(ss, stuff, ' ');
				int y1 = round(atoi(stuff.c_str()) / 10.0); // keep normal for distance/math, flip sign later
				std::getline(ss, stuff, ' ');
				int x2 = round(atoi(stuff.c_str()) / 10.0);
				std::getline(ss, stuff);
				int y2 = round(atoi(stuff.c_str()) / 10.0); // keep normal for distance/math, flip sign later

				double z = ceilingHeight / 2;	// half of height
				double deltaX = x2 - x1;
				double deltaY = y2 - y1;
				double angle = atan2(deltaY, deltaX);
				double midX = (deltaX / 2.0) + x1;
				double midY = (deltaY / 2.0) + y1;
				angle = angle * (180.0 / 3.14);		// convert back to degrees
				double length = sqrt(pow(deltaX, 2) + pow(deltaY, 2));
				midY = -midY;		// flip the sign of Y for UE4 coords
				angle = -angle;		//	flip the sign of Yaw for UE4 coords

				// x y z l w h
				ostringstream os;
				os << "fl " << midX << " " << midY << " " << z << " " << length << " " << wallPadding << " " << angle << " " << ceilingHeight << "\n";
				content += os.str();
			}
			else if (line.substr(0, 16).compare("Cairn: RobotHome") == 0)
			{
				string stuff;
				stringstream ss(line);

				// skip id data
				for (int i = 0; i < 2; i++) std::getline(ss, stuff, ' ');

				std::getline(ss, stuff, ' ');
				int x = round(atoi(stuff.c_str()) / 10.0); // from mm to cm
				std::getline(ss, stuff, ' ');
				int y = round(atoi(stuff.c_str()) / -10.0);
				std::getline(ss, stuff, ' ');
				int heading = round(atof(stuff.c_str()) * -1.0);

				// x y center point and heading
				ostringstream os;
				os << "rh " << x << " " << y << " " << heading << "\n";
				content += os.str();
			}			
		}
		if (line.compare("LINES") == 0)
			foundLINES = true;
	}

	mapFile.close();

	short numPackets = ceil(content.size() / (double)MAX_PACKET_SIZE);
	broadcastPacket.AddToPayload<short>(numPackets);
	for (int i = 0; i < UserClients.size(); i++)
	{
		myServer->SendToClient(&broadcastPacket, UserClients[i]->GetClientEntry());
	}
	broadcastPacket.ClearPayload();

	for (int i = 0; i < (int)numPackets; i++)
	{
		string subStr = content.substr(i * MAX_PACKET_SIZE, MAX_PACKET_SIZE);
		broadcastPacket.AddStringToPayload(subStr);
		for (int j = 0; j < UserClients.size(); j++)
		{
			myServer->SendToClient(&broadcastPacket, UserClients[j]->GetClientEntry());
		}
		broadcastPacket.ClearPayload();
	}
	
	updateScreen("-------------ArLevel File Sent-------------");
	logWrite(SERVER, "ARLevel file sent to all connected clients");
}

void MoveROBITToPoint(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method) // just sending the path
{
	CrossClientID robitID = packet->RemoveFromPayload<CrossClientID>();
	
	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	if (vetoClient && vetoClient->IsROBITClient()) {
		ROBITClient* robitClient = (ROBITClient*)vetoClient;

		for (int j = 0; j < robitClient->fullySubscribedUsers.size(); j++)
		{
			myServer->SendToClient(packet, robitClient->fullySubscribedUsers[j]->GetClientEntry());
		}

		int size = packet->RemoveFromPayload<int>(); // amount of points
		robitClient->path.clear();
		for (int j = 0; j < size; j++)
		{
			Position3D nextPose;
			nextPose.x = packet->RemoveFromPayload<double>();
			nextPose.y = packet->RemoveFromPayload<double>();
			nextPose.z = packet->RemoveFromPayload<double>();

			robitClient->path.push_back(nextPose);
		}

		//No complex logging so we don't block calls
		logWrite(INST, "Path sent to robit: " + to_string(robitID));
	}
}

void ROBITAtGoal(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	if (vetoClient && vetoClient->IsROBITClient()) {
		ROBITClient* robitClient = (ROBITClient*)vetoClient;

		for (int j = 0; j < robitClient->fullySubscribedUsers.size(); j++)
		{
			myServer->SendToClient(packet, robitClient->fullySubscribedUsers[j]->GetClientEntry());
		}

		robitClient->path.clear();

		// log goal reached
		ostringstream os;
		os << "Robit " << robitClient->id << " is at GOAL"; 
		string logS = os.str();
		updateScreen(logS);
		logWrite(INST, logS);
	}
}

void SetROBITFacing(const CrossPack* packet, CrossClientEntryPtr client, NetTransMethod method)
{
	CrossClientID robitID = packet->RemoveFromPayload<CrossClientID>();

	VETOClient* vetoClient = client->GetCustomData<VETOClient>();
	if (vetoClient && vetoClient->IsROBITClient()) {
		ROBITClient* robitClient = (ROBITClient*)vetoClient;

		for (int j = 0; j < robitClient->minimallySubscribedUsers.size(); j++)
		{
			myServer->StreamToClient(packet, robitClient->minimallySubscribedUsers[j]->GetClientEntry());
		}

		double pitch = packet->RemoveFromPayload<float>();
		double yaw = packet->RemoveFromPayload<float>();
		double roll = packet->RemoveFromPayload<float>();

		robitClient->facing = Position3D(0, 0, 0, pitch, yaw, roll);

		// print facings if we care about the position data
		if (verbosePosition)
		{
			ostringstream os;
			os << "Robit [" << robitID << "] is facing: (p/y/r) (" << pitch << ", " << yaw << ", " << roll << ").";
			string logS = os.str();

			cout << logS << endl;
			logWrite(FACE, logS);
		}
	}
}

void RemoveVETOClient(VETOClient* vetoClient)
{
	if (!vetoClient) {

		// log
		logWrite(PROBLEM, "Null disconnect?");

		// update screen
		updateScreen("ERROR: Null disconnect?");

		return;
	}

	string dID = to_string(vetoClient->id);

	if (vetoClient->IsUserClient()) {
		UserClient* userClient = (UserClient*)vetoClient;

		// remove from all ROBITs subscription lists
		for (int j = 0; j < ROBITClients.size(); j++)
		{
			// minimal sub list
			for (int k = 0; k < ROBITClients[j]->minimallySubscribedUsers.size(); k++)
			{
				if (ROBITClients[j]->minimallySubscribedUsers[k]->id == vetoClient->id)
				{
					ROBITClients[j]->minimallySubscribedUsers.erase(ROBITClients[j]->minimallySubscribedUsers.begin() + k);
					break;
				}
			}

			// full sub list
			for (int k = 0; k < ROBITClients[j]->fullySubscribedUsers.size(); k++)
			{
				if (ROBITClients[j]->fullySubscribedUsers[k]->id == vetoClient->id)
				{
					ROBITClients[j]->fullySubscribedUsers.erase(ROBITClients[j]->fullySubscribedUsers.begin() + k);
					break;
				}
			}
		}

		// remove this user from the list
		auto it = UserClients.begin();
		while (it != UserClients.end()) {
			UserClient* user = *it;
			if (user->id == userClient->id) {
				it = UserClients.erase(it);
			}
			else {
				it++;
			}
		}

		string s = "Client disconnect from id: " + dID;

		// log
		logWrite(SERVER, s);

		// update screen
		updateScreen(s);

		return;
	}
	else if (vetoClient->IsROBITClient()) {
		ROBITClient* robitClient = (ROBITClient*)vetoClient;

		// send disconnect packet
		CrossPack Pack;
		Pack.SetDataID(myServer->GetDataIDFromName("ROBITDisconnect"));
		for (int j = 0; j < robitClient->minimallySubscribedUsers.size(); j++)
		{
			myServer->SendToClient(&Pack, robitClient->minimallySubscribedUsers[j]->GetClientEntry());
		}

		// remove from ROBITs list
		auto it = ROBITClients.begin();
		while (it != ROBITClients.end()) {
			ROBITClient* robit = *it;
			if (robit->id == robitClient->id) {
				it = ROBITClients.erase(it);
			}
			else {
				it++;
			}
		}

		// broadcast new ROBIT list to all clients
		CrossPack pack;
		for (int j = 0; j < UserClients.size(); j++)
		{
			GetAvailableROBITs(&pack, UserClients[j]->GetClientEntry(), NetTransMethod::TCP);
		}

		//$$$ This may need to be modified, it fires on robit reconnection and etc... The log looks scary now.
		string s = "Robit disconnect from id: " + dID;

		// log
		logWrite(SERVER, s);

		// update screen
		updateScreen(s);

		return;
	}

	// disconnect from neither a valid robit or client?
	// log
	logWrite(PROBLEM, "Bad Disconnect from neither Robit/Client");

	// update screen
	updateScreen("ERROR! Bad Disconnect from neither Robit/Client?");
}