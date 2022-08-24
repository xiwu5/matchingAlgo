import dataProcessor as dp
import cProfile
import io
import logging
# from memory_profiler import profile
import gc
import sys
import argparse

sys.dont_write_bytecode = True
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('MM')
logger.setLevel(logging.DEBUG)

# Start runtime profiling for main
# @profile(precision=3)
def main():
    parser = argparse.ArgumentParser(description='Match making arg parser.')
    parser.add_argument(
        '--data', help='User data file path', dest='data', default='../simulator/testFakeData.csv')
    parser.add_argument(
        '--rules', help='User data file path', dest='rules', default='../configs/matchingStrategies.json')
    args = parser.parse_args()
    processor = dp.DataProcessor()
    gc.collect()
    # load user data
    processor.loadDataFromFile(args.data, args.rules)
    # apply hard filter
    processor.removeImpossiblePairs()
    gc.collect()
    # apply rating machine and generate total score
    processor.computeRatingsForPairs()
    gc.collect()
    # generate matching pairs
    processor.getPairs()
    processor.generateReports()

if __name__ == '__main__':
    # Run profiler to get an understanding of run-time
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    profilerStats = io.StringIO()
    ps = pstats.Stats(profiler, stream=profilerStats).sort_stats('tottime')
    ps.print_stats()

    # generate profiling report
    with open('../profiling/runTimeProfilingOutput.txt', 'w+') as f:
        f.write(profilerStats.getvalue())