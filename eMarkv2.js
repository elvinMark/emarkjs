/**************/
/*Vector Class*/
/**************/

function eVector(length,str){
	this.length = length;
	this.data = new Array(length);

	if(str != null){
		var temp = str.split(" ");
		for(var i = 0;i<this.length;i++)
			this.data[i] = parseFloat(temp[i]);
	}
	this.toString = function(){
		var s = "";
		for(var i = 0;i<this.length;i++)
			s = s + this.data[i] + " ";
		return s;
	};
	this.copy = function(){
		var out;
		out = new eVector(this.length);
		for(var i = 0;i<this.length;i++)
			out.data[i] = this.data[i];
		return out;
	};
	this.zeros = function(){
		for(var i = 0;i<this.length;i++)
			this.data[i] = 0;
	};
	this.ones = function(){
		for(var i = 0;i<this.length;i++)
			this.data[i] = 1;
	};
	this.random = function(){
		for(var i = 0;i<this.length;i++)
			this.data[i] = Math.random();
	};
	this.add = function(v){
		var out = new eVector(this.length);
		for(var i = 0;i<this.length;i++)
			out.data[i] = this.data[i] + v.data[i];
		return out;
	};
	this.diff = function(v){
		var out = new eVector(this.length);
		for(var i = 0;i<this.length;i++)
			out.data[i] = this.data[i] - v.data[i];
		return out;
	};
	this.times = function(v){
		if(typeof(v)=="number"){
			var out = new eVector(this.length);
			for(var i = 0;i<this.length;i++)
				out.data[i] = this.data[i] * v;
			return out;
		}
		else{
			var out = new eVector(this.length);
			for(var i = 0;i<this.length;i++)
				out.data[i] = this.data[i] * v.data[i];
			return out;
		}
	};
	this.dot = function(v){
		var s = 0;
		for(var i = 0;i<this.length;i++)
			s = s + this.data[i] * v.data[i];
		return s;
	};
	this.norm = function(){
		var s = 0;
		for(var i = 0;i<this.length;i++)
			s = s + this.data[i]*this.data[i];
		return Math.sqrt(s);
	};
	this.normalize = function(){
		var n;
		n = this.norm();
		for(var i = 0;i<this.length;i++)
			this.data[i] = this.data[i]/n;
	};
	this.proj = function(v){
		return v.times(this.dot(v));
	};
}

/**************/
/*Matrix Class*/
/**************/
function eMatrix(rows,cols,str){
	this.rows = rows;
	this.cols = cols;
	this.data = new Array(this.rows);
	if(str != null){
		var a = str.split(";");
		for(var i=0;i<this.rows;i++)
			this.data[i] = new eVector(this.cols,a[i]);
	}
	else{
		for(var i = 0;i<this.rows;i++)
			this.data[i] = new eVector(this.cols);
	}

	this.toString = function(){
		var s = "";
		for(var i = 0;i<this.rows;i++){
			s = s + this.data[i].toString() + "\n";
		}
		return s;
	};
	this.copy = function(){
		var out;
		out = new eMatrix(this.rows,this.cols);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[i].data[j] = this.data[i].data[j];
		}
		return out;
	};
	this.zeros = function(){
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				this.data[i].data[j] = 0;
		}
	};
	this.ones = function(){
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				this.data[i].data[j] = 1;
		}
	};
	this.eye = function(){
		this.zeros();
		for(var i = 0;i<this.rows;i++)
			this.data[i].data[i] = 1;
	};
	this.random = function(){
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				this.data[i].data[j] = Math.random();
		}
	};
	this.add = function(M){
		var out;
		out = new eMatrix(this.rows,this.cols);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[i].data[j] = this.data[i].data[j] + M.data[i].data[j];
		}
		return out;
	};
	this.diff = function(M){
		var out;
		out = new eMatrix(this.rows,this.cols);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[i].data[j] = this.data[i].data[j] - M.data[i].data[j];
		}
		return out;
	};
	this.times = function(M){
		var out;
		var s;
		out = new eMatrix(this.rows,this.cols);
		if(typeof(M) == "number"){
			for(var i = 0;i<this.rows;i++){
				for(var j = 0;j<this.cols;j++)
					out.data[i].data[j] = this.data[i].data[j] * M;
			}
			return out;
		}
		else{
			for(var i = 0;i<this.rows;i++){
				for(var j = 0;j<this.cols;j++)
					out.data[i].data[j] = this.data[i].data[j] * M.data[i].data[j];
			}
			return out;	
		}
	};
	this.dot = function(M){
		var out;
		var s;
		out = new eMatrix(this.rows,M.cols);
			for(var i = 0;i<this.rows;i++){
				for(var j = 0;j<M.cols;j++){
					s = 0;
					for(var k = 0;k<this.cols;k++)
						s = s + this.data[i].data[k] * M.data[k].data[j];
					out.data[i].data[j] = s;
				}
			}
			return out;
	};
	this.transpose = function(){
		var out;
		out = new eMatrix(this.cols,this.rows);
		for(var i = 0;i<this.cols;i++){
			for(var j = 0;j<this.rows;j++)
				out.data[i].data[j] = this.data[j].data[i];
		}
		return out;
	};
	this.LU = function(){
		var lu=[];
		var L,U;
		var i,j,k;
		var s;
		L = new eMatrix(this.rows,this.cols);
		U = new eMatrix(this.rows,this.cols);
		L.ones();
		U.zeros();
		for(i = 0;i<this.rows;i++){
			for(j = 0;j<this.cols;j++){
				s = 0;
				for(k=0;k<i;k++)
					s = s + L.data[i].data[k]*U.data[k].data[j];
				U.data[i].data[j] = this.data[i].data[j] - s;
			}
			for(j = i+1;j<this.cols;j++){
				s = 0;
				for(k=0;k<i;k++)
					s = s + L.data[j].data[k]*U.data[k].data[i];
				L.data[j].data[i] = (this.data[j].data[i] - s)/U.data[i].data[i];
			}
		}
		lu.push(L);
		lu.push(U);
		return lu;
	};
	this.QR = function(){
		var qr=[];
		var Q,R,A;
		var aux;
		var i,j,k;
		var s;
		Q = new eMatrix(this.rows,this.cols);
		R = new eMatrix(this.rows,this.cols);
		Q.zeros();
		R.zeros();
		A = this.transpose();
		aux = new eVector(this.cols);
		for(i = 0;i<this.rows;i++){
			aux.zeros();
			for(j=0;j<i;j++){
				R.data[j].data[i] = A.data[i].dot(Q.data[j]);
				aux = aux.add(Q.data[j].times(R.data[j].data[i]));
			}
			Q.data[i] = A.data[i].diff(aux);
			R.data[i].data[i] = Q.data[i].norm();
			Q.data[i].normalize();
		}
		qr.push(Q.transpose());
		qr.push(R);
		return qr;
	};
	this.det = function(){
		var out = 1;
		var lu = this.LU();
		for(var i = 0;i<this.rows;i++)
			out = out * lu[1].data[i].data[i];
		return out;
	};
	this.inv = function(){
		var out;
		var z;
		var i,j,k;
		var s;
		var lu;
		lu = this.LU();
		out = new eMatrix(this.rows,this.cols);
		z = new eVector(this.rows);
		for(j= 0;j<this.cols;j++){
			for(i = 0;i<this.rows;i++){
				s = 0;
				for(k=0;k<i;k++)
					s = s + lu[0].data[i].data[k]*z.data[k];
				if(i==j)
					z.data[i] = 1 - s;
				else
					z.data[i] = -s;
			}
			for(i=this.rows-1;i>=0;i--){
				s=0;
				for(k=i+1;k<this.rows;k++)
					s = s + lu[1].data[i].data[k]*out.data[k].data[j];
				out.data[i].data[j] = (z.data[i] - s)/lu[1].data[i].data[i];
			}
		}
		return out;
	};
	this.eig = function(N){
		var qr;
		var lambda,eigv;
		var out = [];
		eigv = new eMatrix(this.rows,this.cols);
		eigv.eye();
		lambda = this.copy();
		for(var i = 0;i<N;i++){
			qr = lambda.QR();
			eigv = eigv.dot(qr[0]);
			lambda = qr[1].dot(qr[0]);
		}
		out.push(lambda);
		out.push(eigv);
		return out;
	};
}

/**********************/
/*Complex Number Class*/
/**********************/
function eComplex(real,img){
	this.r = real;
	this.i = img;

	this.toString = function(){
		var s = "";
		s = s + this.r + " + "+ this.i + "i";
		return s;
	}
	this.add = function(c){
		var out = new eComplex(0,0);
		out.r = this.r + c.r;
		out.i = this.i + c.i;
		return out;
	};
	this.diff = function(c){
		var out = new eComplex(0,0);
		out.r = this.r - c.r;
		out.i = this.i - c.i;	
	};
	this.times = function(c){
		var out = new eComplex(0,0);
		out.r = this.r * c.r - this.i * c.i;
		out.i = this.r * c.i + this.i * c.r;
		return out;
	};
	this.conjugate = function(){
		return Math.sqrt(this.r*this.r + this.i*this.i);
	};
	this.norm = function(){
		return Math.sqrt(this.r*this.r + this.i*this.i);
	};
}
function complex_exp(c){
	var out;
	out = new eComplex(0,0);
	out.r = Math.exp(c.r)*Math.cos(c.i);
	out.i = Math.exp(c.r)*Math.sin(c.i);
	return out;
}

function fft(data){
	var out = [];
	var i,N;
	var data_even=[],data_odd=[];
	var fft_even,fft_odd;
	var aux = new eComplex(0,0);
	N = data.length;
	if(data.length == 1){
		out.push(data[0]);
		return out;
	}
	else{
		for(i = 0;i<data.length/2;i++){
			data_even.push(data[2*i]);
			data_odd.push(data[2*i+1]);
		}
		fft_even = fft(data_even);
		fft_odd = fft(data_odd);
		for(i = 0;i<data.length;i++){
			aux.i = 2*Math.PI*i/N;
			out.push(fft_even[i%(N/2)].add(fft_odd[i%(N/2)].times(complex_exp(aux))));
		}
	}
	return out;
}

/******************/
/*Polynomial class*/
/******************/
function ePoly(degree,coeff){
	this.degree = degree;
	this.coeff = [];
	if(coeff == null){
		this.coeff = new eVector(degree+1);
		this.coeff.ones();
	}
	else{
		if(typeof(coeff)=="string")
			this.coeff = new eVector(this.degree+1,coeff);
		else
			this.coeff = coeff;
	}
	this.zeros = function(){
		this.coeff.zeros();
	};
	this.ones = function(){
		this.coeff.ones();
	};
	this.random = function(){
		this.coeff.random();
	};
	this.toString = function(){
		var s = "";
		for(var i = 0;i<this.degree;i++)
			s = s + this.coeff.data[i] + "x^" + i + " + ";
		s = s + this.coeff.data[this.degree] + "x^" + i;
		return s;
	};
	this.eval = function(x){
		if(typeof(x)=="number"){
			var s = 1;
			var out = 0;
			for(var i = 0;i<=this.degree;i++){
				out = out + this.coeff.data[i]*s;
				s = s * x;
			}
			return out;
		}
		else{
			var s = new eVector(x.length);
			var out = new eVector(x.length);
			s.ones();
			out.zeros();
			for(var i = 0;i<=this.degree;i++){
				out = out.add(s.times(this.coeff.data[i]));
				s = s.times(x);
			}
			return out;
		}
	};
	this.derivate = function(){
		var out = new ePoly(this.degree-1);
		for(var i= 1;i<=this.degree;i++)
			out.coeff.data[i-1] = this.coeff.data[i]*i;
		return out;
	};
	this.integrate = function(){
		var out = new ePoly(this.degree+1);
		out.zeros();
		for(var i= 0;i<=this.degree;i++)
			out.coeff.data[i+1] = this.coeff.data[i]/(i+1);
		return out;	
	};
	this.add = function(p){
		var out;
		if(this.degree > p.degree){
			out = new ePoly(this.degree);
			for(var i = 0;i<=this.degree;i++){
				if(i<=p.degree)
					out.coeff.data[i] = this.coeff.data[i] + p.coeff.data[i];
				else
					out.coeff.data[i] = this.coeff.data[i];
			}
		}
		else{
			out = new ePoly(p.degree);
			for(var i = 0;i<=p.degree;i++){
				if(i<=this.degree)
					out.coeff.data[i] = this.coeff.data[i] + p.coeff.data[i];
				else
					out.coeff.data[i] = p.coeff.data[i];
			}
		}
		return out;
	};
	this.diff = function(p){
		var out;
		if(this.degree > p.degree){
			out = new ePoly(this.degree);
			for(var i = 0;i<=this.degree;i++){
				if(i<=p.degree)
					out.coeff.data[i] = this.coeff.data[i] - p.coeff.data[i];
				else
					out.coeff.data[i] = this.coeff.data[i];
			}
		}
		else{
			out = new ePoly(p.degree);
			for(var i = 0;i<=p.degree;i++){
				if(i<=this.degree)
					out.coeff.data[i] = this.coeff.data[i] - p.coeff.data[i];
				else
					out.coeff.data[i] = -p.coeff.data[i];
			}
		}
		return out;
	};
	this.times = function(p){
		if(typeof(p)=="number"){
			var out = new ePoly(this.degree);
			for(var i = 0;i<=this.degree;i++)
					out.coeff.data[i] = p*this.coeff.data[i];
			return out;
		}
		else{
			var out = new ePoly(this.degree + p.degree);
			out.zeros();
			for(var i = 0;i<=this.degree;i++){
				for(var j = 0;j<=p.degree;j++)
					out.coeff.data[i + j] = out.coeff.data[i + j] + this.coeff.data[i]*p.coeff.data[j];
			}
			return out;
		}
	};
}
function polynomial_interpolation(x,y){
	var out = new ePoly(0);
	var aux = new ePoly(1);
	var temp;
	var s;
	out.zeros();
	for(var i = 0;i<x.length;i++){
		s = 1;
		temp = new ePoly(0);
		for(var j = 0;j<x.length;j++){
			if(i!=j){
				aux.coeff.data[0] = -x.data[j];
				temp = temp.times(aux);
				s = s * (x.data[i] - x.data[j]);
			}
		}
		temp = temp.times(y.data[i]/s);
		out = out.add(temp);
	}
	return out;
}

/******************/
/*ODE Solver Class*/
/******************/
function eODE(){
	this.time = [];
	this.N = 0;

	this.mesh = function(time_domain,N){
		var h;
		var t;
		h = (time_domain[1] - time_domain[0])/(N-1);
		this.time = [];
		this.time.push(time_domain[0]);
		t = time_domain[0];
		for(var i = 0;i<N-1;i++){
			t = t + h;
			this.time.push(t);
		}
		this.N = N;
	};
	this.eulerSolver = function(fun_f,x0){
		var out;
		var dt;
		var aux;
		out = new eMatrix(this.N,x0.length);
		out.data[0] = x0.copy();
		t = this.time[0];
		for(var i = 1;i<out.rows;i++){
			aux = fun_f(out.data[i-1],this.time[i-1]);
			dt = this.time[i] - this.time[i-1];
			out.data[i] = out.data[i-1].add(aux.times(dt));
		}
		return out;
	};
}

/*****************************/
/*MultiLayer Perceptron Class*/
/*****************************/
function sigmoid(x){
	var out;
	out = new eMatrix(x.rows,x.cols);
	for(var i = 0;i<x.rows;i++){
		for(var j = 0;j<x.cols;j++)
			out.data[i].data[j] = 1/(1 + Math.exp(-x.data[i].data[j]));
	}
	return out;
}
function eLayer(inputs,outputs){
	this.w = new eMatrix(inputs,outputs);
	this.in = [];
	this.out = [];
	this.delta = [];
	this.n_inputs = inputs;
	this.n_outputs = outputs;
	this.w.random();

	this.forward = function(in_set){
		this.in = in_set.copy();
		this.out = sigmoid(in_set.dot(this.w));
	};
	this.backward = function(err){
		var ones = new eMatrix(err.rows,err.cols);
		ones.ones();
		this.delta = this.out.times(ones.diff(this.out)).times(err);
		return this.delta.dot(this.w.transpose());
	};
	this.update = function(){
		this.w = this.w.diff(this.in.transpose().dot(this.delta));
	};

}
function eMLP(struct){
	this.layers = [];
	this.n_layers = struct.length-1;
	this.struct = struct;
	for(var i = 1;i<struct.length;i++)
		this.layers.push(new eLayer(struct[i-1],struct[i]));
	this.forward = function(in_set){
		var aux;
		aux = in_set;
		for(var i = 0;i<this.n_layers;i++){
			this.layers[i].forward(aux);
			aux = this.layers[i].out;
		}
		return aux.copy();
	};
	this.backward = function(err){
		aux = err;
		for(var i = this.n_layers-1;i>=0;i--){
			aux = this.layers[i].backward(aux);
		}
	};
	this.update = function(){
		for(var i = 0;i<this.n_layers;i++)
			this.layers[i].update();
	};
	this.train = function(in_set,out_set,N){
		var err;
		for(var i = 0;i<N;i++){
			err = this.forward(in_set);
			err = err.diff(out_set);
			this.backward(err);
			this.update();
		}
	};
}