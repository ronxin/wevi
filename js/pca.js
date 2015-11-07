/*
  http://bl.ocks.org/ktaneishi/9499896#pca.js
*/
var PCA = function(){
    this.scale = scale;
    this.pca = pca;

    function mean(X){
        // mean by col
        var T = transpose(X);
        return T.map(function(row){ return d3.sum(row) / X.length; });
    }

    function transpose(X){
        return d3.range(X[0].length).map(function(i){
            return X.map(function(row){ return row[i]; });
        });
    }

    function dot(X,Y){
        return X.map(function(row){
            return transpose(Y).map(function(col){
                return d3.sum(d3.zip(row,col).map(function(v){
                    return v[0]*v[1];
                }));
            });
        });
    }

    function diag(X){
        return d3.range(X.length).map(function(i){
            return d3.range(X.length).map(function(j){ return (i == j) ? X[i] : 0; });
        });
    }

    function zeros(i,j){
        return d3.range(i).map(function(row){
            return d3.range(j).map(function(){ return 0; });
        });
    }

    function trunc(X,d){
        return X.map(function(row){
            return row.map(function(x){ return (x < d) ? 0 : x; });
        });
    }

    function same(X,Y){
        return d3.zip(X,Y).map(function(v){
            return d3.zip(v[0],v[1]).map(function(w){ return w[0] == w[1]; });
        }).map(function(row){
            return row.reduce(function(x,y){ return x*y; });
        }).reduce(function(x,y){ return x*y; });     
    }

    function std(X){
        var m = mean(X);
        return sqrt(mean(mul(X,X)), mul(m,m));
    }

    function sqrt(V){
        return V.map(function(x){ return Math.sqrt(x); });
    }

    function mul(X,Y){
        return d3.zip(X,Y).map(function(v){
            if (typeof(v[0]) == 'number') return v[0]*v[1];
            return d3.zip(v[0],v[1]).map(function(w){ return w[0]*w[1]; });
        });
    }

    function sub(x,y){
        console.assert(x.length == y.length, 'dim(x) == dim(y)');
        return d3.zip(x,y).map(function(v){
            if (typeof(v[0]) == 'number') return v[0]-v[1];
            else return d3.zip(v[0],v[1]).map(function(w){ return w[0]-w[1]; });
        });
    }

    function div(x,y){
        console.assert(x.length == y.length, 'dim(x) == dim(y)');
        return d3.zip(x,y).map(function(v){ return v[0]/v[1]; });
    }

    function scale(X, center, scale){
        // compatible with R scale()
        if (center){
            var m = mean(X);
            X = X.map(function(row){ return sub(row, m); });
        }

        if (scale){
            var s = std(X);
            X = X.map(function(row){ return div(row, s); });
        }
        return X;
    }

    // translated from http://stitchpanorama.sourceforge.net/Python/svd.py
    function svd(A){
        var temp;
        // Compute the thin SVD from G. H. Golub and C. Reinsch, Numer. Math. 14, 403-420 (1970)
        var prec = Math.pow(2,-52) // assumes double prec
        var tolerance = 1.e-64/prec;
        var itmax = 50;
        var c = 0;
        var i = 0;
        var j = 0;
        var k = 0;
        var l = 0;
        
        var u = A.map(function(row){ return row.slice(0); });
        var m = u.length;
        var n = u[0].length;
        
        console.assert(m >= n, 'Need more rows than columns');
        
        var e = d3.range(n).map(function(){ return 0; });
        var q = d3.range(n).map(function(){ return 0; });
        var v = zeros(n,n);
        
        function pythag(a,b){
            a = Math.abs(a)
            b = Math.abs(b)
            if (a > b)
                return a*Math.sqrt(1.0+(b*b/a/a))
            else if (b == 0) 
                return a
            return b*Math.sqrt(1.0+(a*a/b/b))
        }

        // Householder's reduction to bidiagonal form
        var f = 0;
        var g = 0;
        var h = 0;
        var x = 0;
        var y = 0;
        var z = 0;
        var s = 0;
        
        for (i=0; i < n; i++)
        {
            e[i]= g;
            s= 0.0;
            l= i+1;
            for (j=i; j < m; j++) 
                s += (u[j][i]*u[j][i]);
            if (s <= tolerance)
                g= 0.0;
            else
            {
                f= u[i][i];
                g= Math.sqrt(s);
                if (f >= 0.0) g= -g;
                h= f*g-s
                u[i][i]=f-g;
                for (j=l; j < n; j++)
                {
                    s= 0.0
                    for (k=i; k < m; k++) 
                        s += u[k][i]*u[k][j]
                    f= s/h
                    for (k=i; k < m; k++) 
                        u[k][j]+=f*u[k][i]
                }
            }
            q[i]= g
            s= 0.0
            for (j=l; j < n; j++) 
                s= s + u[i][j]*u[i][j]
            if (s <= tolerance)
                g= 0.0
            else
            {
                f= u[i][i+1]
                g= Math.sqrt(s)
                if (f >= 0.0) g= -g
                h= f*g - s
                u[i][i+1] = f-g;
                for (j=l; j < n; j++) e[j]= u[i][j]/h
                for (j=l; j < m; j++)
                {
                    s=0.0
                    for (k=l; k < n; k++) 
                        s += (u[j][k]*u[i][k])
                    for (k=l; k < n; k++) 
                        u[j][k]+=s*e[k]
                }
            }
            y= Math.abs(q[i])+Math.abs(e[i])
            if (y>x) 
                x=y
        }
        
        // accumulation of right hand gtransformations
        for (i=n-1; i != -1; i+= -1)
        {
            if (g != 0.0)
            {
                h= g*u[i][i+1]
                for (j=l; j < n; j++) 
                    v[j][i]=u[i][j]/h
                for (j=l; j < n; j++)
                {
                    s=0.0
                    for (k=l; k < n; k++) 
                        s += u[i][k]*v[k][j]
                    for (k=l; k < n; k++) 
                        v[k][j]+=(s*v[k][i])
                }
            }
            for (j=l; j < n; j++)
            {
                v[i][j] = 0;
                v[j][i] = 0;
            }
            v[i][i] = 1;
            g= e[i]
            l= i
        }
        
        // accumulation of left hand transformations
        for (i=n-1; i != -1; i+= -1)
        {
            l= i+1
            g= q[i]
            for (j=l; j < n; j++) 
                u[i][j] = 0;
            if (g != 0.0)
            {
                h= u[i][i]*g
                for (j=l; j < n; j++)
                {
                    s=0.0
                    for (k=l; k < m; k++) s += u[k][i]*u[k][j];
                    f= s/h
                    for (k=i; k < m; k++) u[k][j]+=f*u[k][i];
                }
                for (j=i; j < m; j++) u[j][i] = u[j][i]/g;
            }
            else
                for (j=i; j < m; j++) u[j][i] = 0;
            u[i][i] += 1;
        }
        
        // diagonalization of the bidiagonal form
        prec= prec*x
        for (k=n-1; k != -1; k+= -1)
        {
            for (var iteration=0; iteration < itmax; iteration++)
            {// test f splitting
                var test_convergence = false
                for (l=k; l != -1; l+= -1)
                {
                    if (Math.abs(e[l]) <= prec){
                        test_convergence= true
                        break 
                    }
                    if (Math.abs(q[l-1]) <= prec)
                        break 
                }
                if (!test_convergence){
                    // cancellation of e[l] if l>0
                    c= 0.0
                    s= 1.0
                    var l1= l-1
                    for (i =l; i<k+1; i++)
                    {
                        f= s*e[i]
                        e[i]= c*e[i]
                        if (Math.abs(f) <= prec)
                            break
                        g= q[i]
                        h= pythag(f,g)
                        q[i]= h
                        c= g/h
                        s= -f/h
                        for (j=0; j < m; j++)
                        {
                            y= u[j][l1]
                            z= u[j][i]
                            u[j][l1] =  y*c+(z*s)
                            u[j][i] = -y*s+(z*c)
                        } 
                    }
                }
                // test f convergence
                z= q[k]
                if (l== k){
                    //convergence
                    if (z<0.0)
                    { //q[k] is made non-negative
                        q[k]= -z
                        for (j=0; j < n; j++)
                            v[j][k] = -v[j][k]
                    }
                    break  //break out of iteration loop and move on to next k value
                }

                console.assert(iteration < itmax-1, 'Error: no convergence.');

                // shift from bottom 2x2 minor
                x= q[l]
                y= q[k-1]
                g= e[k-1]
                h= e[k]
                f= ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
                g= pythag(f,1.0)
                if (f < 0.0)
                    f= ((x-z)*(x+z)+h*(y/(f-g)-h))/x
                else
                    f= ((x-z)*(x+z)+h*(y/(f+g)-h))/x
                // next QR transformation
                c= 1.0
                s= 1.0
                for (i=l+1; i< k+1; i++)
                {
                    g = e[i]
                    y = q[i]
                    h = s*g
                    g = c*g
                    z = pythag(f,h)
                    e[i-1] = z
                    c = f/z
                    s = h/z
                    f = x*c+g*s
                    g = -x*s+g*c
                    h = y*s
                    y = y*c
                    for (j =0; j < n; j++)
                    {
                        x = v[j][i-1]
                        z = v[j][i]
                        v[j][i-1]  = x*c+z*s
                        v[j][i]  = -x*s+z*c
                    }
                    z = pythag(f,h)
                    q[i-1] = z
                    c = f/z
                    s = h/z
                    f = c*g+s*y
                    x = -s*g+c*y
                    for (j =0; j < m; j++)
                    {
                        y = u[j][i-1]
                        z = u[j][i]
                        u[j][i-1]  = y*c+z*s
                        u[j][i]  = -y*s+z*c
                    }
                }
                e[l] = 0.0
                e[k] = f
                q[k] = x
            } 
        }
            
        // vt = transpose(v)
        // return (u,q,vt)
        for (i=0;i<q.length; i++) 
            if (q[i] < prec) q[i] = 0
          
        // sort eigenvalues
        for (i=0; i< n; i++){ 
            // writeln(q)
            for (j=i-1; j >= 0; j--){
                if (q[j] < q[i]){
                    // writeln(i,'-',j)
                    c = q[j]
                    q[j] = q[i]
                    q[i] = c
                    for (k=0;k<u.length;k++) { temp = u[k][i]; u[k][i] = u[k][j]; u[k][j] = temp; }
                    for (k=0;k<v.length;k++) { temp = v[k][i]; v[k][i] = v[k][j]; v[k][j] = temp; }
                    i = j   
                }
            }
        }
        return { U:u, S:q, V:v }
    }

    function pca(X,npc){
        var USV = svd(X);
        var U = USV.U;
        var S = diag(USV.S);
        var V = USV.V;

        // T = X*V = U*S
        var pcXV = dot(X,V)
        var pcUdS = dot(U,S);

        var prod = trunc(sub(pcXV,pcUdS), 1e-12);
        var zero = zeros(prod.length, prod[0].length);
        console.assert(same(prod,zero), 'svd and eig ways must be the same.');
        
        return pcUdS;
    }
};