#ifndef RDMA_QUEUE_PAIR_H
#define RDMA_QUEUE_PAIR_H

#include <ns3/object.h>
#include <ns3/packet.h>
#include <ns3/ipv4-address.h>
#include <ns3/data-rate.h>
#include <ns3/event-id.h>
#include <ns3/custom-header.h>
#include <ns3/int-header.h>
#include <vector>

namespace ns3 {

class RdmaQueuePair : public Object {
public:
	Time startTime;
	Ipv4Address sip, dip;
	uint16_t sport, dport;
	uint64_t m_size;
	uint64_t snd_nxt, snd_una; // next seq to send, the highest unacked seq
	uint16_t m_pg;
	uint16_t m_ipid;
	uint32_t m_win; // bound of on-the-fly packets
	uint64_t m_baseRtt; // base RTT of this qp
	DataRate m_max_rate; // max rate
	bool m_var_win; // variable window size
	Time m_nextAvail;	//< Soonest time of next send
	uint32_t wp; // current window of packets
	uint32_t lastPktSize;
	Callback<void> m_notifyAppFinish;

	/******************************
	 * runtime states
	 *****************************/
	DataRate m_rate;	//< Current rate
	struct {
		DataRate m_targetRate;	//< Target rate
		EventId m_eventUpdateAlpha;
		double m_alpha;
		bool m_alpha_cnp_arrived; // indicate if CNP arrived in the last slot
		bool m_first_cnp; // indicate if the current CNP is the first CNP
		EventId m_eventDecreaseRate;
		bool m_decrease_cnp_arrived; // indicate if CNP arrived in the last slot
		uint32_t m_rpTimeStage;
		EventId m_rpTimer;
	} mlx;
	struct {
		uint32_t m_lastUpdateSeq;
		DataRate m_curRate;
		IntHop hop[IntHeader::maxHop];
		uint32_t keep[IntHeader::maxHop];
		uint32_t m_incStage;
		double m_lastGap;
		double u;
		struct {
			double u;
			DataRate Rc;
			uint32_t incStage;
		}hopState[IntHeader::maxHop];
	} hp;
	struct{
		uint32_t m_lastUpdateSeq;
		DataRate m_curRate;
		uint32_t m_incStage;
		uint64_t lastRtt;
		double rttDiff;
	} tmly;
	struct{
		uint32_t m_lastUpdateSeq;
		uint32_t m_caState;
		uint32_t m_highSeq; // when to exit cwr
		double m_alpha;
		uint32_t m_ecnCnt;
		uint32_t m_batchSizeOfAlpha;
	} dctcp;
	struct{
		uint32_t m_lastUpdateSeq;
		DataRate m_curRate;
		uint32_t m_incStage;
	}hpccPint;
	struct{
		// Fusion regulation control parameters (from RL agent)
		double delta;      // 0.5 <= delta <= 1.5
		double gamma;      // 0.5 <= gamma <= 1.5
		double beta;       // 0 < beta < 1
		double mu1;        // 2 < mu1 <= mu2 <= 4
		double mu2;        // 2 < mu1 <= mu2 <= 4
		double mu3;        // 6 <= mu3 <= 10
		
		// Fusion state variables
		DataRate m_targetRate;  // R_T: target rate
		DataRate m_curRate;     // R_C: current rate
		DataRate m_refRate;     // R_REF: reference rate
		uint64_t m_lastRtt;     // last RTT measurement
		uint64_t m_minRtt;      // minimum RTT observed
		double m_alpha;         // alpha for rate decrease
		uint32_t m_timer;       // T: timer counter
		uint32_t m_byteCounter; // BC: byte counter
		uint64_t m_byteCounterAccumulator; // Accumulated bytes for BC increment
		EventId m_timerEvent;   // Timer event for periodic T increment
		uint32_t m_lastUpdateSeq;
		Time m_lastUpdateTime;  // last monitoring interval update
		bool m_cnpReceived;     // CNP received flag
		
		// Statistics for loss and buffer occupancy collection
		uint32_t m_ecnMarkedPackets;  // Number of ECN-marked packets (qfb)
		uint32_t m_totalPackets;      // Total number of packets (total)
		uint32_t m_nackCount;         // Number of NACKs received (indicates loss)
		uint32_t m_packetsSent;       // Total packets sent (for loss calculation)
		uint32_t m_packetsAcked;      // Total packets acknowledged
		Time m_lastStatsReset;        // Last time statistics were reset
	} fusion;

	/***********
	 * methods
	 **********/
	static TypeId GetTypeId (void);
	RdmaQueuePair(uint16_t pg, Ipv4Address _sip, Ipv4Address _dip, uint16_t _sport, uint16_t _dport);
	void SetSize(uint64_t size);
	void SetWin(uint32_t win);
	void SetBaseRtt(uint64_t baseRtt);
	void SetVarWin(bool v);
	void SetAppNotifyCallback(Callback<void> notifyAppFinish);

	uint64_t GetBytesLeft();
	uint32_t GetHash(void);
	void Acknowledge(uint64_t ack);
	uint64_t GetOnTheFly();
	bool IsWinBound();
	uint64_t GetWin(); // window size calculated from m_rate
	bool IsFinished();
	uint64_t HpGetCurWin(); // window size calculated from hp.m_curRate, used by HPCC
};

class RdmaRxQueuePair : public Object { // Rx side queue pair
public:
	struct ECNAccount{
		uint16_t qIndex;
		uint8_t ecnbits;
		uint16_t qfb;
		uint16_t total;

		ECNAccount() { memset(this, 0, sizeof(ECNAccount));}
	};
	ECNAccount m_ecn_source;
	uint32_t sip, dip;
	uint16_t sport, dport;
	uint16_t m_ipid;
	uint32_t ReceiverNextExpectedSeq;
	Time m_nackTimer;
	int32_t m_milestone_rx;
	uint32_t m_lastNACK;
	EventId QcnTimerEvent; // if destroy this rxQp, remember to cancel this timer

	static TypeId GetTypeId (void);
	RdmaRxQueuePair();
	uint32_t GetHash(void);
};

class RdmaQueuePairGroup : public Object {
public:
	std::vector<Ptr<RdmaQueuePair> > m_qps;
	//std::vector<Ptr<RdmaRxQueuePair> > m_rxQps;

	static TypeId GetTypeId (void);
	RdmaQueuePairGroup(void);
	uint32_t GetN(void);
	Ptr<RdmaQueuePair> Get(uint32_t idx);
	Ptr<RdmaQueuePair> operator[](uint32_t idx);
	void AddQp(Ptr<RdmaQueuePair> qp);
	//void AddRxQp(Ptr<RdmaRxQueuePair> rxQp);
	void Clear(void);
};

}

#endif /* RDMA_QUEUE_PAIR_H */
