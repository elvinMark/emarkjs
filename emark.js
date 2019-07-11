///////////////////////////
/* Author: Elvin Mark MV */
///////////////////////////


////////////////////
/* Linear Algebra */
////////////////////

function init_vector(elems){
	var data;
	data = new Array(elems);
	for(var i = 0;i<elems;i++){
		data[i] = 0;
	}
	var v = {
		n : elems,
		data : data
	};
	return v;
}

function init_matrix(rows,cols){
	var data = [];
	var aux;
	for(var i = 0;i<rows;i++){
		aux = init_vector(cols);
		data.push(aux);
	}
	var M = {
		rows : rows,
		cols : cols,
		data : data
	};
	return M;
}

function ones_vector(elems){
	var data;
	data = new Array(elems);
	for(var i = 0;i<elems;i++){
		data[i] = 1;
	}
	var v = {
		n : elems,
		data : data
	};
	return v;
}

function ones_matrix(rows,cols){
	var data = [];
	var aux;
	for(var i = 0;i<rows;i++){
		aux = ones_vector(cols);
		data.push(aux);
	}
	var M = {
		rows : rows,
		cols : cols,
		data : data
	};
	return M;
}

function unit_vector(elems,pos){
	var data;
	data = new Array(elems);
	for(var i = 0;i<elems;i++){
		data[i] = 0;
	}
	data[pos] = 1;
	var v = {
		n : elems,
		data : data
	};
	return v;	
}

function identity_matrix(dim){
	var data = [];
	var aux;
	for(var i = 0;i<dim;i++){
		aux = unit_vector(dim,i);
		data.push(aux);
	}
	var M = {
		rows : dim,
		cols : dim,
		data : data
	};
	return M;
}

function random_vector(elems){
	var data;
	data = new Array(elems);
	for(var i = 0;i<elems;i++){
		data[i] = Math.random();
	}
	var v = {
		n : elems,
		data : data
	};
	return v;
}

function random_matrix(rows,cols){
	var data = [];
	var aux;
	for(var i = 0;i<rows;i++){
		aux = random_vector(cols);
		data.push(aux);
	}
	var M = {
		rows : rows,
		cols : cols,
		data : data
	};
	return M;
}

function str_2_vector(str,elems){
	var data;
	var aux;
	aux = str.split(" ");
	data = new Array(elems);
	for(var i =0;i<elems;i++){
		data[i] = parseFloat(aux[i]);
	}
	var v = {
		n : elems,
		data : data
	};
	return v;

}

function str_2_matrix(str,rows,cols){
	var data = [];
	var aux = str.split(";");
	for(var i = 0;i<rows;i++){
		data.push(str_2_vector(aux[i],cols));
	}
	var M = {
		rows : rows,
		cols : cols,
		data : data
	};
	return M;
}
// Functions to print in the console

function print_vector(v){
	var s = "";
	for(var i = 0;i<v.n;i++){
		s = s + v.data[i]+ " ";
	}
	console.log(s);
}

function print_matrix(M){
	var s = "";
	for(var i=0;i<M.rows;i++){
		for(var j=0;j<M.cols;j++){
			s = s + M.data[i].data[j] + " ";
		}
		s = s + "\n";
	}
	console.log(s);
}

// Basic operations with vector

function sum_vector(v1,v2){
	var out = init_vector(v1.n);
	for(var i = 0;i<v1.n;i++){
		out.data[i] = v1.data[i] + v2.data[i];
	}
	return out;
}

function diff_vector(v1,v2){
	var out = init_vector(v1.n);
	for(var i = 0;i<v1.n;i++){
		out.data[i] = v1.data[i] - v2.data[i];
	}
	return out;
}

function dot_vector(v1,v2){
	var out = 0;
	for(var i = 0;i<v1.n;i++){
		out = out + v1.data[i]*v2.data[i];
	}
	return out;
}

function prod_vector(l,v1){
	var out = init_vector(v1.n);
	for(var i = 0;i<v1.n;i++){
		out.data[i] = l*v1.data[i];
	}
	return out;	
}

function norm_vector(v1){
	var out=0;
	for(var i=0;i<v1.n;i++){
		out = out + v1.data[i]*v1.data[i]
	}
	return Math.sqrt(out);
}

// Basic Opeartions with matrix

function sum_matrix(m1,m2){
	var out = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[i].data[j] = m1.data[i].data[j]+m2.data[i].data[j];
		}
	}
	return out;
}

function diff_matrix(m1,m2){
	var out = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[i].data[j] = m1.data[i].data[j]-m2.data[i].data[j];
		}
	}
	return out;
}

function dot_matrix(m1,m2){
	var out = init_matrix(m1.rows,m2.cols);
	var s;
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m2.cols;j++){
			s = 0;
			for(var k =0;k<m1.cols;k++){
				s = s + m1.data[i].data[k]*m2.data[k].data[j]
			}
			out.data[i].data[j] = s;
		}
	}
	return out;
}

function prod_matrix(m1,m2){
	var out = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[i].data[j] = m1.data[i].data[j]*m2.data[i].data[j];
		}
	}
	return out;
}

function prod_num_matrix(l,m1){
	var out = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[i].data[j] = l*m1.data[i].data[j];
		}
	}	
	return out;
}

function prod_matrix_vector(m,v){
	var out = init_vector(m.rows);
	var s ;
	for(var i=0;i<m.rows;i++){
		s = 0;
		for(var j=0;j<m.cols;j++){
			s = s + m.data[i].data[j]*v.data[j];
		}
		out.data[i] = s;
	}
	return out;
}

function trans_matrix(m1){
	var out = init_matrix(m1.cols,m1.rows);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			out.data[j].data[i] = m1.data[i].data[j];
		}
	}	
	return out;	
}

function lu_fact_matrix(m1){
	var L,U;
	var out = [];
	var s;
	L = identity_matrix(m1.rows,m1.cols);
	U = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j=i;j<m1.cols;j++){
			s = 0;
			for(var k=0;k<j;k++){
				s = s + L.data[i].data[k]*U.data[k].data[j];
			}
			U.data[i].data[j] = (m1.data[i].data[j] - s);
		}
		for(var j = i+1;j<m1.cols;j++){
			s = 0;
			for(var k = 0;k<i;k++){
				s = s + L.data[j].data[k]*U.data[k].data[i];
			}
			L.data[j].data[i] = (m1.data[j].data[i] - s)/U.data[i].data[i];
		}
	}
	out.push(L);
	out.push(U);
	return out;
}

function det_matrix(m1){
	var out = 1;
	var lu;
	lu = lu_fact_matrix(m1);
	for(var i = 0;i<m1.rows;i++){
		out = out * lu[1].data[i].data[i];
	}
	return out;
}

function inverse_matrix(m1){
	var out = init_matrix(m1.rows,m1.cols);
	var z = init_vector(m1.rows);
	var lu;
	var s;

	lu = lu_fact_matrix(m1);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.rows;j++){
			s = 0;
			for(var k = 0;k<j;k++){
				s = s + lu[0].data[j].data[k]*z.data[k]; 
			}
			if(i==j){
				z.data[j] = 1 - s;
			}
			else{
				z.data[j] = -s;
			} 
		}
		for(var j = m1.rows-1;j>=0;j--){
			s = 0;
			for(var k=j+1;k<m1.rows;k++){
				s = s + lu[1].data[j].data[k]*out.data[k].data[i];
			}
			out.data[j].data[i] = (z.data[j] - s)/lu[1].data[j].data[j];
		}
	}
	return out;
}

function qr_fact_matrix(m1){
	var Q,R;
	var out = [];
	var aux;
	var m = trans_matrix(m1);
	Q = init_matrix(m1.rows,m1.cols);
	R = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m.rows;i++){
		aux = init_vector(m.rows);
		for(var j = 0;j<i;j++){
			aux = sum_vector(aux,prod_vector(dot_vector(m.data[i],Q.data[j]),Q.data[j]));
		}
		aux = diff_vector(m.data[i],aux);
		R.data[i].data[i] = norm_vector(aux);
		aux = prod_vector(1/R.data[i].data[i],aux);
		Q.data[i] = aux;
		for(var j=0;j<i;j++){
			R.data[i].data[j] = dot_vector(m.data[i],Q.data[j]);
		}
	}
	out.push(trans_matrix(Q));
	out.push(trans_matrix(R));
	return out;
}

function eigen_matrix(m1,N){
	var qr;
	var out;
	var out_vec;
	out = m1;
	out_vec = identity_matrix(m1.rows);
	for(var i = 0;i<N;i++){
		qr  = qr_fact_matrix(out);
		out = dot_matrix(qr[1],qr[0]);
		out_vec = dot_matrix(out_vec,qr[0]);
	}
	return [out,out_vec];
}

function solve_linear_system(m,b){
	var z = init_vector(b.n);
	var out = init_vector(b.n);
	var lu = lu_fact_matrix(m);
	var s;
	
	for(var j = 0;j<b.n;j++){
		s = 0;
		for(var k = 0;k<j;k++){
			s = s + lu[0].data[j].data[k]*z.data[k];
		}
		z.data[j] = b.data[j] - s;
	}
	for(var j = b.n-1;j>=0;j--){
		s = 0;
		for(var k = j+1;k<b.n;k++){
			s = s + lu[1].data[j].data[k]*out.data[k];
		}
		out.data[j] = (z.data[j] - s)/lu[1].data[j].data[j];
	}
	return out;
}

/////////////////////
/* Complex Numbers */
/////////////////////

function init_complex(r,i){
	var out = {
		re : r,
		im : i
	};
	return out;
}

function sum_complex(c1,c2){
	var out = init_complex(c1.re + c2.re,c1.im + c2.im);
	return out;
}

function diff_complex(c1,c2){
	var out = init_complex(c1.re - c2.re,c1.im - c2.im);
	return out;	
}

function prod_complex(c1,c2){
	var out = init_complex(c1.re*c2.re - c1.im*c2.im,c1.im*c2.re + c1.re*c2.im);
	return out;
}

function conjugate_complex(c1){
	return init_complex(c1.re,-c1.im);
}

function norm_complex(c1){
	return Math.sqrt(c1.re*c1.re + c1.im*c1.im);
}

function arg_complex(c1){
	return Math.atan(c1.im/c1.re);
}

function exp_complex(c1){
	var out = init_complex(Math.exp(c1.re)*Math.cos(c1.im),Math.exp(c1.re)*Math.sin(c1.im));
	return out;
}

function fft(data){
	if (data.length == 1){
		return [data[0]];
	}
	var data_o = [];
	var data_e = [];
	var N = data.length;
	for(var i = 0;i<N;i++){
		if(i%2){
			data_o.push(data[i]);
		}
		else{
			data_e.push(data[i]);
		}
	}
	var x1 = fft(data_o);
	var x2 = fft(data_e);
	var out = [];
	for(var k = 0;k<N;k++){
		out.push(sum_complex(x2[k%(N/2)],prod_complex(x1[k%(N/2)],exp_complex(init_complex(0,2*Math.PI*k/N)))));
	}
	return out;
}

//////////////////
/*Neural Network*/
//////////////////

//Simple Multilayer Perceptron using sigmoid as activation function

function sigmoid_matrix(m1){
	var o;
	o = init_matrix(m1.rows,m1.cols);
	for(var i = 0;i<m1.rows;i++){
		for(var j = 0;j<m1.cols;j++){
			o.data[i].data[j] = 1.0/(1+Math.exp(-m1.data[i].data[j]));
		}
	}
	return o;
}

function init_layer(inputs,outputs){
	var l = {
		n_inputs : inputs,
		n_outputs : outputs,
		w : random_matrix(inputs,outputs),
		forward : function(inp){
			var o;
			o = sigmoid_matrix(dot_matrix(inp,this.w));
			this.i = inp;
			this.o = o;
		},
		print : function(){
			print_matrix(this.w);
		},
		backward : function(err){
			var ones = ones_matrix(err.rows,err.cols);
			var aux = prod_matrix(this.o,diff_matrix(ones,this.o));
			this.delta = prod_matrix(err,aux);
			return dot_matrix(this.delta,trans_matrix(this.w));
		},
		update : function(alpha){
			this.w = diff_matrix(this.w,prod_num_matrix(alpha,dot_matrix(trans_matrix(this.i),this.delta)));
		}
	};
	return l;
}

function init_mlp(structure){
	var l = [];
	for(var i = 0;i<structure.length-1;i++){
		l.push(init_layer(structure[i],structure[i+1]));
	}
	var n = {
		struct : structure,
		n_layers : structure.length-1,
		layers : l,
		forward : function(inp){
			var o;
			o = inp;
			for(var i=0;i<this.n_layers;i++){
				this.layers[i].forward(o);
				o = this.layers[i].o;
			}
			this.o = o;
		},
		print : function(){
			for(var i = 0;i<this.n_layers;i++){
				this.layers[i].print();
			}
		},
		backward : function(err){
			var aux = err;
			for(var i = this.n_layers-1;i>=0;i--){
				aux = this.layers[i].backward(aux);
			}
		},
		update : function(alpha){
			for(var i = 0;i<this.n_layers;i++){
				aux = this.layers[i].update(alpha);
			}	
		},
		train : function(in_set,out_set,alpha,N){
			var o;
			for(var i = 0;i<N;i++){
				this.forward(in_set);
				this.backward(diff_matrix(this.o,out_set));
				this.update(alpha);
			}
		}

	};
	return n;
}

/////////////////////////////
/*Solve non-linear equation*/
/////////////////////////////

function newton_method(fun_f,fun_df,x0,N){
	var x;
	x = x0;
	for(var i = 0;i < N;i++){
		x = x - fun_f(x)/fun_df(x0);
	}
	return x;
}

function bisection_method(fun_f,a,b,N){
	var xa;
	var xb;
	var mid;
	xa = a;
	xb = b;
	for(var i = 0;i<N;i++){
		mid = (xa + xb)/2;
		if(fun_f(xa)*fun_f(mid) <0){
			xb = mid;
		}
		else{
			xa = mid;
		}
	}
	return xa;
}

//////////////
/* Calculus */
//////////////

function derivate_function(fun_f,x0,h){
	return (fun_f(x0+h) - fun_f(x0))/h;
}

function integral_function(fun_f,a,b,N){
	var h = (b-a)/N;
	var s = 0;
	var x = a;
	for(var i = 0;i<N;i++){
		s = s + fun_f(x)*h;
		x = x + h;
	}
	return h;
}

function ode_solver(fun_f,x0,a,b,N){
	var h = (b-a)/N;
	var t = a;
	var out_x = [];
	var out_t = []
	var aux,dx;
	out_x.push(x0);
	out_t.push(t);
	for(var i = 1;i<N;i++){
		dx = fun_f(t,out_x[i-1]);
		dx = prod_vector(h,dx);
		aux = sum_vector(out_x[i-1],dx);
		t = t + h;
		out_x.push(aux);
		out_t.push(t);
	}
	return [out_t,out_x];
}

function ode_runge_kutta(fun_f,x0,a,b,N){
	var h = (b-a)/N;
	var t = a;
	var k1 = init_vector(x0.length);
	var k2 = init_vector(x0.length);
	var k3 = init_vector(x0.length);
	var k4 = init_vector(x0.length);
	var out_x = [];
	var out_t = [];
	out_x.push(x0);
	out_t.push(t);
	for(var i = 1;i<N;i++){
		k1 = fun_f(t,out_x[i-1]);
		k1 = prod_vector(h,k1);
		
		k2 = prod_vector(0.5,k1);
		k2 = sum_vector(out_x[i-1],k2);
		k2 = fun_f(t+h/2,k2);
		k2 = prod_vector(h,k2);
		
		k3 = prod_vector(0.5,k2);
		k3 = sum_vector(out_x[i-1],k3);
		k3 = fun_f(t+h/2,k3);
		k3 = prod_vector(h,k3);

		k4 = sum_vector(out_x[i-1],k3);
		k4 = fun_f(t+h,k4);
		k4 = prod_vector(h,k4);
		
		t = t + h;
		k2 = prod_vector(2,k2);
		k3 = prod_vector(2,k3);
		k1 = sum_vector(k1,k2);
		k1 = sum_vector(k1,k3);
		k1 = sum_vector(k1,k4);
		k1 = prod_vector(1.0/6.0,k1);
		k1 = sum_vector(out_x[i-1],k1);

		out_t.push(t);
		out_x.push(k1);
	}
	return [out_t,out_x];
}

////////////////
/* Polynomial */
////////////////

function init_polynomial(degree){
	var p = new Array(degree + 1);
	for(var i = 0;i<=degree;i++){
		p[i] = 0;
	}
	return p;
}

function eval_polynomial(p,x){
	var s=0;
	var x0=1;
	for(var i = 0;i<p.length;i++){
		s = s + p[i]*x0;
		x0 = x0*x;
	}
	return s;
}

function prod_polynomial(p1,p2){
	var p;
	var n1,n2;
	n1 = p1.length;
	n2 = p2.length;
	p = init_polynomial(n1 + n2 - 2);
	for(var i = 0;i<n1;i++){
		for(var j = 0;j<n2;j++){
			p[i + j] = p[i + j] + p1[i]*p2[j];
		}
	}
	return p;
}
function sum_polynomial(p1,p2){
	var out ;
	var n1,n2,n;
	n1 = p1.length;
	n2 = p2.length;
	n = n1 > n2? n1 : n2;
	out = new Array(n);
	if(n1 > n2){
		for(var i = 0;i<n2;i++){
			out[i] = p1[i] + p2[i];
		}
		for(var i = n2; i<n1;i++){
			out[i] = p1[i];
		}
	}
	else{
		for(var i = 0;i<n1;i++){
			out[i] = p1[i] + p2[i];
		}
		for(var i = n1; i<n2;i++){
			out[i] = p2[i];
		}
	}
	return out;
}

function prod_num_polynomial(num,p1){
	var out;
	var n;
	n = p1.length;
	out = new Array(n);
	for(var i = 0;i<n;i++){
		out[i] = num*p1[i];
	}
	return out;
}

/////////////\//////
/* Interpolation */
///////////////////

function lagrange_interpolation(data){
	var out =[];
	var n;
	var p;
	var s;
	n = data.length;
	out = init_polynomial(n-1);
	for(var i = 0;i<n;i++){
		p = [1,0];
		s = 1;
		for(var j = 0;j<n;j++){
			if(i!=j){
				p = prod_polynomial(p,[-data[j][0],1]);
				s = s*(data[i][0]-data[j][0]);
			}
		}
		p=prod_num_polynomial(data[i][1]/s,p);
		out = sum_polynomial(out,p);
	}
	return out;
}


////////////////
/* Statistics */
////////////////

function average(v){
	var out;
	out = 0;
	for(var i =0;i<v.n;i++){
		out = out + v.data[i];
	}
	out = out/v.n;
	return out;
}

function std_deviation(v){
	var avg = average(v);
	var out = 0;
	for(var i = 0;i < v.n;i++){
		out = out + (v.data[i] - avg)*(v.data[i] - avg);
	}
	out = Math.sqrt(out/v.n);
	return out;
}

function combinatory(n,k){
	if(k==0 || k == n)
		return 1;
	return combinatory(n-1,k) + combinatory(n-1,k-1);
}

function factorial(n){
	if(n==0 || n==1)
		return 1;
	return n*factorial(n-1);
}

function linear_regression(x_data,y_data){
	var A = init_matrix(2,2);
	var b = init_vector(2);
	var sx = 0;
	var sy = 0;
	for(var i =0;i<x_data.n;i++){
		sx = sx + x_data.data[i];
	}
	for(var i =0;i<y_data.n;i++){
		sy = sy + y_data.data[i];
	}
	A.data[0].data[0] = dot_vector(x_data,x_data);
	A.data[0].data[1] = sx;
	A.data[1].data[0] = sx;
	A.data[1].data[1] = x_data.n;
	b.data[0] = dot_vector(x_data,y_data);
	b.data[1] = sy;
	print_matrix(A);
	print_vector(b);
	return solve_linear_system(A,b);
}

function gaussian_distribution(x,avg,std){
	return (1/Math.sqrt(2*Math.PI*std*std))*Math.exp(-Math.pow(x-avg,2)/(2*Math.pow(std,2)));
}

function binomial_distribution(x,p,N){
	return combinatory(N,x)*Math.pow(p,x)*Math.pow(1-p,N-x);
}

///////////////////
/* Graphing Tool */
///////////////////

function init_graph(context,WIDTH,HEIGHT,xlim,ylim){
	var out = {
		ctx : context,
		h : HEIGHT,
		w : WIDTH,
		xlim : xlim,
		ylim : ylim,
		hx : WIDTH/(xlim[1] - xlim[0]),
		hy : HEIGHT/(ylim[1] - ylim[0]),
		draw_background : function(Nx,Ny){
			var hx = this.w/Nx;
			var hy = this.h/Ny;
			this.ctx.beginPath();
			this.ctx.strokeStyle = "#000000";
			this.ctx.lineWidth = 0.5;
			for(var i = 0;i<Nx;i++){
				this.ctx.moveTo(i*hx,0);
				this.ctx.lineTo(i*hx,this.h);
			}
			for(var i =0;i<Ny;i++){
				this.ctx.moveTo(0,i*hy);
				this.ctx.lineTo(this.h,i*hy);	
			}
			this.ctx.stroke();
			this.ctx.beginPath();
			this.ctx.strokeStyle = "#CC0000";
			this.ctx.lineWidth = 2;
			this.ctx.moveTo(0,Ny*hy/2);
			this.ctx.lineTo(this.w,Ny*hy/2);
			this.ctx.moveTo(Nx*hx/2,0);
			this.ctx.lineTo(Nx*hx/2,this.h);
			this.ctx.stroke();
			this.ctx.beginPath();
			this.ctx.strokeStyle = "#000000";
			this.ctx.font ="7px Arial";
			for(var i = 0;i<Nx;i++){
				this.ctx.moveTo(i*hx,this.h/2-5);
				this.ctx.lineTo(i*hx,this.h/2+5);
				this.ctx.fillText(((i*hx-this.w/2)/this.hx).toFixed(2),i*hx-5,this.h/2+20);
			}
			for(var i = 0;i<Ny;i++){
				this.ctx.moveTo(this.w/2-5,i*hy);
				this.ctx.lineTo(this.w/2+5,i*hy);
				this.ctx.fillText(((i*hy-this.h/2)/this.hy).toFixed(2),this.w/2-40,i*hy+5);
			}
			this.ctx.stroke();
			
		},
		draw_arrow : function(r0,rf,color){
			var v1 = rf[0] - r0[0];
			var v2 = rf[1] - r0[1];
			var d = Math.sqrt(v1*v1 + v2*v2);
			var u1 = -v1/d;
			var u2 = -v2/d;
			var p11 = rf[0] + (d/10)*(u1*Math.cos(Math.PI/7) - u2*Math.sin(Math.PI/7));
			var p12 = rf[1] + (d/10)*(u1*Math.sin(Math.PI/7) + u2*Math.cos(Math.PI/7));
			var p21 = rf[0] + (d/10)*(u1*Math.cos(-Math.PI/7) - u2*Math.sin(-Math.PI/7));
			var p22 = rf[1] + (d/10)*(u1*Math.sin(-Math.PI/7) + u2*Math.cos(-Math.PI/7));
			this.ctx.lineWidth = 2;
			this.ctx.strokeStyle = color;
			this.ctx.beginPath();
			this.ctx.moveTo(this.w/2 + r0[0]*this.hx,this.h/2 - r0[1]*this.hy);
			this.ctx.lineTo(this.w/2 + rf[0]*this.hx,this.h/2 - rf[1]*this.hy);
			this.ctx.lineTo(this.w/2 + p11*this.hx,this.h/2 - p12*this.hy);
			this.ctx.lineTo(this.w/2 + p21*this.hx,this.h/2 - p22*this.hy);
			this.ctx.lineTo(this.w/2 + rf[0]*this.hx,this.h/2 - rf[1]*this.hy);
			this.ctx.fillStyle = color;
			this.ctx.fill();
			this.ctx.stroke();
		},
		plot : function(datax,datay,color){
			var N = datax.length;
			this.ctx.lineWidth = 1;
			this.ctx.strokeStyle = color;
			this.ctx.beginPath();
			this.ctx.moveTo(this.w/2 + this.hx*datax[0],this.h/2 - this.hy*datay[0]);
			for(var i = 1;i<N;i++){
				this.ctx.lineTo(this.w/2 + this.hx*datax[i],this.h/2 - this.hy*datay[i]);
			}
			this.ctx.stroke();
		},
		clear : function(){
			this.ctx.clearRect(0,0,this.w,this.h);
		}
	};
	return out;
}
