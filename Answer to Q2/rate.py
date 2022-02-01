import math
import numpy as np
import matplotlib.pyplot as plt


def de(es, e, s):
    """
    The rate of change for E
    :param es: initial concentration of ES
    :param e: initial concentration of E
    :param s: initial concentration of S
    :return: k2 * ES - k1 * E * S + k3 * ES
    """
    k1 = 100
    k2 = 600
    k3 = 150

    return k2 * es - k1 * e * s + k3 * es


def ds(es, e, s):
    """
    The rate of change for S
    :param es: initial concentration of ES
    :param e: initial concentration of E
    :param s: initial concentration of S
    :return: k2 * ES - k1 * E * S
    """
    k1 = 100
    k2 = 600

    return k2 * es - k1 * e * s


def des(es, e, s):
    """
    The rate of change for ES
    :param es: initial concentration of ES
    :param e: initial concentration of E
    :param s: initial concentration of S
    :return: k1 * E * S - k2 * ES - k3 * ES
    """
    k1 = 100
    k2 = 600
    k3 = 150

    return k1 * e * s - k2 * es - k3 * es


def dp(es):
    """
    The rate of change for P
    :param es: initial concentration of ES
    :return: k3 * ES
    """
    k3 = 150

    return k3 * es


def RK4(e, s, es, p, h):
    # first step
    e1 = de(es, e, s)
    m1 = e + e1 * h * 0.5

    s1 = ds(es, e, s)
    n1 = s + s1 * h * 0.5

    es1 = des(es, e, s)
    o1 = es + es1 * h * 0.5

    p1 = dp(es)
    q1 = p + p1 * h * 0.5

    # second step
    e2 = de(o1, m1, n1)
    m2 = e + e2 * h * 0.5

    s2 = ds(o1, m1, n1)
    n2 = s + s2 * h * 0.5

    es2 = des(o1, m1, n1)
    o2 = es + es2 * h * 0.5

    p2 = dp(o1)
    q2 = p + p2 * h * 0.5

    # third step
    e3 = de(o2, m2, n2)
    m3 = e + e3 * h

    s3 = ds(o2, m2, n2)
    n3 = s + s3 * h

    es3 = des(o2, m2, n2)
    o3 = es + es3 * h

    p3 = dp(o2)
    q3 = p + p3 * h

    # forth step
    e4 = de(o3, m3, n3)

    s4 = ds(o3, m3, n3)

    es4 = des(o3, m3, n3)

    p4 = dp(o3)

    # return
    e = e + (e1 + 2 * e2 + 2 * e3 + e4) * h / 6
    s = s + (s1 + 2 * s2 + 2 * s3 + s4) * h / 6
    es = es + (es1 + 2 * es2 + 2 * es3 + es4) * h / 6
    p = p + (p1 + 2 * p2 + 2 * p3 + p4) * h / 6

    return e, s, es, p


def main():
    # INPUT
    # inital concentration
    e = 1
    s = 10
    es = 0
    p = 0
    k3 = 150
    v = k3 * es

    t = 0       # time
    h = 0.00001     # step

    # array
    t_array = []
    e_array = []
    s_array = []
    es_array = []
    p_array = []
    v_array = []

    print('{}\t\t{}\t\t\t{}\t\t\t{}\t\t\t{}'.format("#", "E", "S", "ES", "P"))
    print('{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(0, e, s, es, p))
    i = 0
    while t < 1:
        i += 1
        t_array.append(t)
        e_array.append(e)
        s_array.append(s)
        es_array.append(es)
        p_array.append(p)
        v_array.append(v)

        t += h
        e, s, es, p = RK4(e, s, es, p, h)
        v = k3 * es
        if i <= 10:
            print('{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(i, e, s, es, p))

    # Plot Concentration changes
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Changes of the 4 concentration")
    axs[0, 0].plot(t_array, e_array)
    axs[0, 0].set_title("concentration of E")
    axs[0, 1].plot(t_array, s_array, 'tab:orange')
    axs[0, 1].set_title("concentration of S")
    axs[1, 0].plot(t_array, es_array, 'tab:green')
    axs[1, 0].set_title("concentration of ES")
    axs[1, 1].plot(t_array, p_array, 'tab:red')
    axs[1, 1].set_title("concentration of P")
    for ax in axs.flat:
        ax.set(xlabel='time (min)', ylabel='concentration (μM)')

    # Plot V as a function of concentration of S
    # V is the rate of change of P
    fig, ax = plt.subplots()
    ax.plot(s_array, v_array)
    ax.annotate('Vm=82.648 μM/min', xy=(350,295), xycoords='figure points')
    ax.set_title('the velocity V as a function of the concentration of the substrate S')
    ax.set(xlabel='concentration of S (μM)', ylabel='rate of change for P (μM/min)')

    plt.show()

    print("\nVm is {:.3f} μM/min".format(max(v_array)))


if __name__ == "__main__":
    main()





