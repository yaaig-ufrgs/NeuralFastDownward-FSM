(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear a)
		(clear b)
		(handempty)
		(on a m)
		(on b e)
		(on c g)
		(on d i)
		(on f d)
		(on g k)
		(on h l)
		(on i n)
		(on j c)
		(on k f)
		(on l j)
		(on m h)
		(ontable e)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
