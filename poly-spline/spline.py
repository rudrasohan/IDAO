def fit2(t,x,vx):
    t_new = np.array(t)/1000
    x_new = np.array(x)/1000
    vx_new = np.array(vx)
    A = []
    y = []
    for i,time in enumerate(t_new):
        A.append([time**2,time,1])
        A.append([2*(time**1),1,0])
        y.append(x_new[i])
        y.append(vx_new[i])
    A = np.array(A)
    y = np.array(y)
    #print(A.shape,y.shape)
    a = (np.linalg.inv(A.T.dot(A))).dot(A.T.dot(y))
    print(np.mean((A.dot(a)-y)**2))
    return a
