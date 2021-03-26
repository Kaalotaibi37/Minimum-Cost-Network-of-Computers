import main as m
import time


for number_of_times in range(5, 55, 5):
    average_p = 0
    best_p = 0
    worst_P = 0
    average_k = 0
    best_k = 0
    worst_k = 0

    for x in range(20):

        # calcuting time complexity for prim algorithm
        # initialization graph
        g = m.Graph(number_of_times)
        prim_algothim = m.Prim(g)

        # calculate time
        start_time_p = time.perf_counter()
        prim_algothim.prim()
        end_time_p = time.perf_counter()
        time_total = (end_time_p - start_time_p)*1000
        # check for the best complexity
        if time_total < best_p or x == 0:
            best_p = time_total
        # check for the worst complexity
        if time_total > worst_P or x == 0:
            worst_P = time_total
        average_p += time_total

        # calcuting time complexity for krystal's  algorithm
        # initialization graph
        g = m.Graph(number_of_times)
        kruskal_algothim = m.Kruskal(g)

        # calculate time
        start_time_k = time.perf_counter()
        kruskal_algothim.kruskal()
        end_time_k = time.perf_counter()
        time_total_k = (end_time_k - start_time_k)*1000
        # check for the best complexity
        if time_total_k < best_k or x == 0:
            best_k = time_total_k
        # check for the worst complexity
        if time_total_k > worst_k or x == 0:
            worst_k = time_total_k
        average_k += time_total_k

    # calculate the average
    average_p /= 20
    average_k /= 20

    print("Prim algorithm \n Number of nodes : {number_of_times} \n best time complexity: {best_p} in millisecond \n worst time complexity: {worst_P} in millisecond \n avarge time complexity: {average_p} in millisecond".format(
        number_of_times=number_of_times, best_p=best_p, worst_P=worst_P, average_p=average_p))
    print("krystal algorithm \n Number of nodes : {number_of_times} \n best time complexity: {best_k} in millisecond \n worst time complexity: {worst_k} in millisecond \n avarge time complexity: {average_k} in millisecond".format(
        number_of_times=number_of_times, best_k=best_k, worst_k=worst_k, average_k=average_k))
    print("========================================================================")
