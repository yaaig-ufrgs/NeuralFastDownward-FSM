(define (problem blocks-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init
	(clear a)
	(clear b)
	(clear c)
	(handempty)
	(on a f)
	(on b d)
	(on c e)
	(on e g)
	(ontable d)
	(ontable f)
	(ontable g)
)
(:goal (and (on a g) (on g d) (on d b) (on b c) (on c f) (on f e)))
)