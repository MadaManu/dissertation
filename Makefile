.PHONY: clean All

All:
	@echo "----------Building project:[ masters_dissertation - Debug ]----------"
	@$(MAKE) -f  "masters_dissertation.mk"
clean:
	@echo "----------Cleaning project:[ masters_dissertation - Debug ]----------"
	@$(MAKE) -f  "masters_dissertation.mk" clean
