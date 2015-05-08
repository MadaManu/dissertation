#include <iostream>
#include <string>
#include <config.h>

using namespace std;

template <class T>
void test_2ptr(T, string message, string resultsdir, string filename, string pcmfile, T (*benchmark)(T*, T*))
{
  cout << message << endl;
  // declaration
  T * a = new T[SIZE];
  T * b = new T[SIZE];

  // initialisation
  populate_array(a);
  populate_array(b);

  uint64_t start, stop;
  uint64_t diffs[runs];
  double instsPerClock[runs];
  double l2HitRate[runs];
  double l3HitRate[runs];

  PCM * m = PCM::getInstance();
  if (m->program() != PCM::Success) exit(1);

  SystemCounterState before_sstate, after_sstate;

  for (int i = 1; i < runs+1; i++) {
    before_sstate = getSystemCounterState();
    start = clocks();
    benchmark(a, b);
    stop = clocks();
    after_sstate = getSystemCounterState();
    diffs[i-1] = stop - start;
    instsPerClock[i-1]  = getIPC(before_sstate,after_sstate);
    l2HitRate[i-1]      = getL2CacheHitRatio(before_sstate,after_sstate);
    l3HitRate[i-1]      = getL3CacheHitRatio(before_sstate,after_sstate);

    // repopulate
    populate_array(a);
    populate_array(b);
  }

  m->cleanup();

  char mkcmd[256];
  sprintf(mkcmd, "mkdir -p %s/", resultsdir.c_str());
  if( system(mkcmd) < 0 ){
    cout << "ERROR occured making folder: " << resultsdir;
    exit(1);
  }

  char timespath[256], ipcpath[256], l2hitpath[256], l3hitpath[256];
  ofstream timestream, ipcstream, l2hitstream, l3hitstream;
  sprintf(timespath, "%s/%s-cycles.txt", resultsdir.c_str(), filename.c_str());
  sprintf(ipcpath,   "%s/%s-ipc.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l2hitpath, "%s/%s-l2h.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l3hitpath, "%s/%s-l3h.txt", resultsdir.c_str(), pcmfile.c_str());

  timestream.open(timespath);
  ipcstream.open(ipcpath);
  l2hitstream.open(l2hitpath);
  l3hitstream.open(l3hitpath);

  for(int k = 0; k < runs; k++) {
    timestream  << diffs[k]         << endl;
    ipcstream   << instsPerClock[k] << endl;
    l2hitstream << l2HitRate[k]     << endl;
    l3hitstream << l3HitRate[k]     << endl;
  }

  timestream.flush();ipcstream.flush();l2hitstream.flush();l3hitstream.flush();
  timestream.close();ipcstream.close();l2hitstream.close();l3hitstream.close();

  cout << "Done. Results in " << resultsdir << endl;
}


template <class T>
void test_ptr_scalar(T, string message, string resultsdir, string filename, string pcmfile, void (*benchmark)(T*, T))
{
  cout << message << endl;
  // declaration
  T * a = new T[SIZE];
  T b;

  // initialisation
  populate_array(a);
  srand(5);
  b = (T) (rand() % 1024);

  uint64_t start, stop;
  uint64_t diffs[runs];
  double instsPerClock[runs];
  double l2HitRate[runs];
  double l3HitRate[runs];

  PCM * m = PCM::getInstance();
  if (m->program() != PCM::Success) exit(1);

  SystemCounterState before_sstate, after_sstate;

  for (int i = 1; i < runs+1; i++) {
    before_sstate = getSystemCounterState();
    start = clocks();
    benchmark(a, b);
    stop = clocks();
    after_sstate = getSystemCounterState();
    diffs[i-1] = stop - start;
    instsPerClock[i-1]  = getIPC(before_sstate,after_sstate);
    l2HitRate[i-1]      = getL2CacheHitRatio(before_sstate,after_sstate);
    l3HitRate[i-1]      = getL3CacheHitRatio(before_sstate,after_sstate);

    // repopulate
    populate_array(a);
    b = (T) (rand() % 1024);
  }

  m->cleanup();

  char mkcmd[256];
  sprintf(mkcmd, "mkdir -p %s/", resultsdir.c_str());
  if( system(mkcmd) < 0 ){
    cout << "ERROR occured making folder: " << resultsdir;
    exit(1);
  }

  char timespath[256], ipcpath[256], l2hitpath[256], l3hitpath[256];
  ofstream timestream, ipcstream, l2hitstream, l3hitstream;
  sprintf(timespath, "%s/%s-cycles.txt", resultsdir.c_str(), filename.c_str());
  sprintf(ipcpath,   "%s/%s-ipc.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l2hitpath, "%s/%s-l2h.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l3hitpath, "%s/%s-l3h.txt", resultsdir.c_str(), pcmfile.c_str());

  timestream.open(timespath);
  ipcstream.open(ipcpath);
  l2hitstream.open(l2hitpath);
  l3hitstream.open(l3hitpath);

  for(int k = 0; k < runs; k++) {
    timestream  << diffs[k]         << endl;
    ipcstream   << instsPerClock[k] << endl;
    l2hitstream << l2HitRate[k]     << endl;
    l3hitstream << l3HitRate[k]     << endl;
  }

  timestream.flush();ipcstream.flush();l2hitstream.flush();l3hitstream.flush();
  timestream.close();ipcstream.close();l2hitstream.close();l3hitstream.close();

  cout << "Done. Results in " << resultsdir << endl;
}

template <class T>
void test_ptr(T, string message, string resultsdir, string filename, string pcmfile, T (*benchmark)(T*))
{
  cout << message << endl;
  // declaration
  T * a = new T[SIZE];
  T b;

  // initialisation
  populate_array(a);

  uint64_t start, stop;
  uint64_t diffs[runs];
  double instsPerClock[runs];
  double l2HitRate[runs];
  double l3HitRate[runs];

  PCM * m = PCM::getInstance();
  if (m->program() != PCM::Success) exit(1);

  SystemCounterState before_sstate, after_sstate;

  for (int i = 1; i < runs+1; i++) {
    before_sstate = getSystemCounterState();
    start = clocks();
    benchmark(a);
    stop = clocks();
    after_sstate = getSystemCounterState();
    diffs[i-1] = stop - start;
    instsPerClock[i-1]  = getIPC(before_sstate,after_sstate);
    l2HitRate[i-1]      = getL2CacheHitRatio(before_sstate,after_sstate);
    l3HitRate[i-1]      = getL3CacheHitRatio(before_sstate,after_sstate);

    // repopulate
    populate_array(a);
  }

  m->cleanup();

  char mkcmd[256];
  sprintf(mkcmd, "mkdir -p %s/", resultsdir.c_str());
  if( system(mkcmd) < 0 ){
    cout << "ERROR occured making folder: " << resultsdir;
    exit(1);
  }

  char timespath[256], ipcpath[256], l2hitpath[256], l3hitpath[256];
  ofstream timestream, ipcstream, l2hitstream, l3hitstream;
  sprintf(timespath, "%s/%s-cycles.txt", resultsdir.c_str(), filename.c_str());
  sprintf(ipcpath,   "%s/%s-ipc.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l2hitpath, "%s/%s-l2h.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l3hitpath, "%s/%s-l3h.txt", resultsdir.c_str(), pcmfile.c_str());

  timestream.open(timespath);
  ipcstream.open(ipcpath);
  l2hitstream.open(l2hitpath);
  l3hitstream.open(l3hitpath);

  for(int k = 0; k < runs; k++) {
    timestream  << diffs[k]         << endl;
    ipcstream   << instsPerClock[k] << endl;
    l2hitstream << l2HitRate[k]     << endl;
    l3hitstream << l3HitRate[k]     << endl;
  }

  timestream.flush();ipcstream.flush();l2hitstream.flush();l3hitstream.flush();
  timestream.close();ipcstream.close();l2hitstream.close();l3hitstream.close();

  cout << "Done. Results in " << resultsdir << endl;
}

template <class T>
void test_mat_vec(T, string message, string resultsdir, string filename, string pcmfile, void (*benchmark)(T**, T*&))
{
  cout << message << endl;
  // declaration
  T ** a = new T* [SIZE];
  T * b = new T [SIZE];

  // initialisation
  populate_matrix(a);
  populate_array(b);

  uint64_t start, stop;
  uint64_t diffs[runs];
  double instsPerClock[runs];
  double l2HitRate[runs];
  double l3HitRate[runs];

  PCM * m = PCM::getInstance();
  if (m->program() != PCM::Success) exit(1);

  SystemCounterState before_sstate, after_sstate;

  for (int i = 1; i < runs+1; i++) {
    before_sstate = getSystemCounterState();
    start = clocks();
    benchmark(a, b);
    stop = clocks();
    after_sstate = getSystemCounterState();
    diffs[i-1] = stop - start;
    instsPerClock[i-1]  = getIPC(before_sstate,after_sstate);
    l2HitRate[i-1]      = getL2CacheHitRatio(before_sstate,after_sstate);
    l3HitRate[i-1]      = getL3CacheHitRatio(before_sstate,after_sstate);

    // repopulate
    populate_matrix(a);
    populate_array(b);
  }

  m->cleanup();

  char mkcmd[256];
  sprintf(mkcmd, "mkdir -p %s/", resultsdir.c_str());
  if( system(mkcmd) < 0 ){
    cout << "ERROR occured making folder: " << resultsdir;
    exit(1);
  }

  char timespath[256], ipcpath[256], l2hitpath[256], l3hitpath[256];
  ofstream timestream, ipcstream, l2hitstream, l3hitstream;
  sprintf(timespath, "%s/%s-cycles.txt", resultsdir.c_str(), filename.c_str());
  sprintf(ipcpath,   "%s/%s-ipc.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l2hitpath, "%s/%s-l2h.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l3hitpath, "%s/%s-l3h.txt", resultsdir.c_str(), pcmfile.c_str());

  timestream.open(timespath);
  ipcstream.open(ipcpath);
  l2hitstream.open(l2hitpath);
  l3hitstream.open(l3hitpath);

  for(int k = 0; k < runs; k++) {
    timestream  << diffs[k]         << endl;
    ipcstream   << instsPerClock[k] << endl;
    l2hitstream << l2HitRate[k]     << endl;
    l3hitstream << l3HitRate[k]     << endl;
  }

  timestream.flush();ipcstream.flush();l2hitstream.flush();l3hitstream.flush();
  timestream.close();ipcstream.close();l2hitstream.close();l3hitstream.close();

  cout << "Done. Results in " << resultsdir << endl;
}

template <class T>
void test_mat_mat(T, string message, string resultsdir, string filename, string pcmfile, T** (*benchmark)(T**, T**))
{
  cout << message << endl;
  // declaration
  T ** a = new T* [SIZE];
  T ** b = new T* [SIZE];

  // initialisation
  populate_matrix(a);
  populate_matrix(b);

  uint64_t start, stop;
  uint64_t diffs[runs];
  double instsPerClock[runs];
  double l2HitRate[runs];
  double l3HitRate[runs];

  PCM * m = PCM::getInstance();
  if (m->program() != PCM::Success) exit(1);

  SystemCounterState before_sstate, after_sstate;

  for (int i = 1; i < runs+1; i++) {
    before_sstate = getSystemCounterState();
    start = clocks();
    benchmark(a, b);
    stop = clocks();
    after_sstate = getSystemCounterState();
    diffs[i-1] = stop - start;
    instsPerClock[i-1]  = getIPC(before_sstate,after_sstate);
    l2HitRate[i-1]      = getL2CacheHitRatio(before_sstate,after_sstate);
    l3HitRate[i-1]      = getL3CacheHitRatio(before_sstate,after_sstate);

    // repopulate
    populate_matrix(a);
    populate_matrix(b);
  }

  m->cleanup();

  char mkcmd[256];
  sprintf(mkcmd, "mkdir -p %s/", resultsdir.c_str());
  if( system(mkcmd) < 0 ){
    cout << "ERROR occured making folder: " << resultsdir;
    exit(1);
  }

  char timespath[256], ipcpath[256], l2hitpath[256], l3hitpath[256];
  ofstream timestream, ipcstream, l2hitstream, l3hitstream;
  sprintf(timespath, "%s/%s-cycles.txt", resultsdir.c_str(), filename.c_str());
  sprintf(ipcpath,   "%s/%s-ipc.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l2hitpath, "%s/%s-l2h.txt", resultsdir.c_str(), pcmfile.c_str());
  sprintf(l3hitpath, "%s/%s-l3h.txt", resultsdir.c_str(), pcmfile.c_str());

  timestream.open(timespath);
  ipcstream.open(ipcpath);
  l2hitstream.open(l2hitpath);
  l3hitstream.open(l3hitpath);

  for(int k = 0; k < runs; k++) {
    timestream  << diffs[k]         << endl;
    ipcstream   << instsPerClock[k] << endl;
    l2hitstream << l2HitRate[k]     << endl;
    l3hitstream << l3HitRate[k]     << endl;
  }

  timestream.flush();ipcstream.flush();l2hitstream.flush();l3hitstream.flush();
  timestream.close();ipcstream.close();l2hitstream.close();l3hitstream.close();

  cout << "Done. Results in " << resultsdir << endl;
}
