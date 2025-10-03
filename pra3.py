#pra3 : A* Search


import queue as Q
from RMP import dict_gn
from RMP import dict_hn

start = 'Arad'
goal = 'Bucharest'
result = ''


def get_fn(citystr):
    cities = citystr.split(",")
    cities = [c.strip() for c in cities]  # remove spaces
    gn = 0
    for ctr in range(len(cities) - 1):
        gn += dict_gn[cities[ctr]][cities[ctr + 1]]
    hn = dict_hn[cities[-1]]
    return hn + gn


def expand(cityq):
    global result
    visited = set()

    while not cityq.empty():
        tot, citystr, thiscity = cityq.get()

        if thiscity in visited:
            continue
        visited.add(thiscity)

        if thiscity == goal:
            result = citystr + " :: " + str(tot)
            return

        for cty in dict_gn[thiscity]:
            newpath = citystr + " , " + cty
            fn = get_fn(newpath)
            cityq.put((fn, newpath, cty))


def main():
    cityq = Q.PriorityQueue()
    cityq.put((get_fn(start), start, start))
    expand(cityq)
    print("The A* path with the total is: ")
    print(result)


main()
