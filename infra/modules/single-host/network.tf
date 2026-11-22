# One public subnet whose only route out is the internet gateway; no NAT gateway. The instance's network interface
# carries the Elastic IP, its address for its own calls out (the image registry, Hugging Face, the Anthropic API,
# Systems Manager, the package repositories) and CloudFront's origin. Its security group admits one flow in: HTTP on
# port 80 from CloudFront's origin-facing addresses. No port 22: access is through Systems Manager only.

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = local.name
  }
}

# The VPC's default security group admits nothing (it is attached to nothing here).
resource "aws_default_security_group" "this" {
  vpc_id = aws_vpc.this.id
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = local.name
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.this.id
  availability_zone       = local.zone
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, 0)
  map_public_ip_on_launch = false # the Elastic IP below is the instance's only public address

  tags = {
    Name = "${local.name}-public-${local.zone}"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${local.name}-public"
  }
}

resource "aws_route" "internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.this.id
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# ---- The instance's security group: one rule per resource (a group created here starts with no rule at all) ----

# CloudFront's origin-facing addresses, as AWS publishes and maintains them.
data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

resource "aws_security_group" "host" {
  name        = "${local.name}-host"
  description = "The single host: HTTP from CloudFront origin-facing addresses only"
  vpc_id      = aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "http_from_cloudfront" {
  security_group_id = aws_security_group.host.id
  description       = "HTTP from CloudFront origin-facing addresses"
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
  prefix_list_id    = data.aws_ec2_managed_prefix_list.cloudfront.id
}

resource "aws_vpc_security_group_egress_rule" "out" {
  security_group_id = aws_security_group.host.id
  description       = "Out: image registry, Hugging Face, Anthropic API, Systems Manager, package repositories, DNS"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

# ---- The address: on its own network interface, so the instance has it from its first second of boot ----

resource "aws_network_interface" "host" {
  subnet_id       = aws_subnet.public.id
  security_groups = [aws_security_group.host.id]
  description     = "${local.name}: the instance's primary interface, with the Elastic IP"

  tags = {
    Name = local.name
  }
}

# Its public DNS name is CloudFront's origin (cdn.tf): it stays when the instance is replaced.
resource "aws_eip" "host" {
  domain            = "vpc"
  network_interface = aws_network_interface.host.id

  tags = {
    Name = local.name
  }

  # an address in a VPC can be associated only once the VPC has an internet gateway
  depends_on = [aws_internet_gateway.this]
}
